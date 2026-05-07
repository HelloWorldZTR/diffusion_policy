"""LegendVLA-compatible WebSocket server for Diffusion Policy checkpoints."""
from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import asyncio
from contextlib import nullcontext
import http
import logging
import pathlib
import time
import traceback
from typing import Any, Dict, Iterable, Mapping

import numpy as np

import diffusion_policy.serving.msgpack_numpy as msgpack_numpy

logger = logging.getLogger(__name__)


def _to_plain_container(value: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return value


def _get_nested(value: Any, path: Iterable[str], default: Any = None) -> Any:
    current = value
    for key in path:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _resize_hwc_nearest(image: np.ndarray, height: int, width: int) -> np.ndarray:
    in_height, in_width = image.shape[:2]
    if (in_height, in_width) == (height, width):
        return image
    y_idx = np.clip(np.round(np.linspace(0, in_height - 1, height)).astype(np.int64), 0, in_height - 1)
    x_idx = np.clip(np.round(np.linspace(0, in_width - 1, width)).astype(np.int64), 0, in_width - 1)
    return image[y_idx][:, x_idx]


def _resize_hwc(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    try:
        import cv2

        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    except Exception:
        pass
    try:
        from PIL import Image

        return np.asarray(Image.fromarray(image).resize((width, height), Image.BILINEAR))
    except Exception:
        return _resize_hwc_nearest(image, height, width)


class LegendVlaObservationAdapter:
    """Adapt LegendVLA serving payloads to Diffusion Policy observation tensors."""

    def __init__(
        self,
        shape_meta: Mapping[str, Any],
        n_obs_steps: int,
        device: str | None = None,
        image_key: str = "image",
        states_key: str = "states",
        head_camera_name: str = "head",
        chest_camera_name: str = "chest",
    ) -> None:
        self.shape_meta = _to_plain_container(shape_meta)
        self.n_obs_steps = int(n_obs_steps)
        self.device = device
        self.image_key = image_key
        self.states_key = states_key
        self.head_camera_name = head_camera_name
        self.chest_camera_name = chest_camera_name

        obs_meta = self.shape_meta["obs"]
        self.image_shape = tuple(int(x) for x in obs_meta["image"]["shape"])
        self.breast_image_shape = None
        if "breast_image" in obs_meta:
            self.breast_image_shape = tuple(int(x) for x in obs_meta["breast_image"]["shape"])
        self.state_dim = int(obs_meta["state"]["shape"][0])
        self.action_dim = int(self.shape_meta["action"]["shape"][0])

    @property
    def requires_chest_view(self) -> bool:
        return self.breast_image_shape is not None

    def _select_camera_value(self, value: Any, camera_name: str) -> Any:
        if isinstance(value, Mapping):
            if camera_name in value:
                return value[camera_name]
            if camera_name == self.chest_camera_name and "breast" in value:
                return value["breast"]
            raise ValueError(f"LegendVLA image payload is missing camera {camera_name!r}")
        if self.requires_chest_view:
            raise ValueError("LegendVLA image payload must be a camera dict with head and chest views")
        return value

    def _as_thwc(self, value: Any, key: str) -> np.ndarray:
        array = np.asarray(value)
        if array.ndim == 3:
            if array.shape[-1] in (3, 4):
                array = array[None, ...]
            elif array.shape[0] in (3, 4):
                array = np.transpose(array[:3], (1, 2, 0))[None, ...]
            else:
                raise ValueError(f"{key} must be HWC/CHW or THWC/TCHW, got shape {array.shape}")
        elif array.ndim == 4:
            if array.shape[-1] in (3, 4):
                pass
            elif array.shape[1] in (3, 4):
                array = np.transpose(array[:, :3], (0, 2, 3, 1))
            else:
                raise ValueError(f"{key} must be HWC/CHW or THWC/TCHW, got shape {array.shape}")
        else:
            raise ValueError(f"{key} must be HWC/CHW or THWC/TCHW, got shape {array.shape}")

        if array.shape[-1] == 4:
            array = array[..., :3]
        return array

    def _pad_or_truncate_history(self, array: np.ndarray, horizon: int) -> np.ndarray:
        if array.shape[0] >= horizon:
            return array[-horizon:]
        if array.shape[0] == 0:
            raise ValueError("history must contain at least one frame")
        pad = np.repeat(array[:1], horizon - array.shape[0], axis=0)
        return np.concatenate([pad, array], axis=0)

    def _prepare_image_history(self, value: Any, key: str, image_shape: tuple[int, int, int]) -> np.ndarray:
        channels, height, width = image_shape
        if channels != 3:
            raise ValueError(f"{key} expects 3-channel RGB image shape, got {image_shape}")

        array = self._as_thwc(value, key)
        array = self._pad_or_truncate_history(array, self.n_obs_steps)
        frames = []
        for frame in array:
            frame = _resize_hwc(frame, height, width)
            frame = frame.astype(np.float32, copy=False)
            if frame.max(initial=0.0) > 1.5:
                frame = frame / 255.0
            frames.append(np.transpose(frame, (2, 0, 1)))
        return np.stack(frames, axis=0).astype(np.float32, copy=False)

    def _prepare_state_history(self, value: Any) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2:
            raise ValueError(f"{self.states_key} must be D or TD, got shape {array.shape}")
        if array.shape[-1] != self.state_dim:
            raise ValueError(f"{self.states_key} last dimension must be {self.state_dim}, got {array.shape[-1]}")
        return self._pad_or_truncate_history(array, self.n_obs_steps).astype(np.float32, copy=False)

    def prepare_numpy_obs(self, obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        image_value = obs.get(self.image_key)
        if image_value is None:
            raise ValueError(f"missing required field {self.image_key!r}")
        state_value = obs.get(self.states_key)
        if state_value is None:
            raise ValueError(f"missing required field {self.states_key!r}")

        head_image = self._select_camera_value(image_value, self.head_camera_name)
        result = {
            "image": self._prepare_image_history(head_image, "image", self.image_shape),
            "state": self._prepare_state_history(state_value),
        }
        if self.breast_image_shape is not None:
            chest_image = self._select_camera_value(image_value, self.chest_camera_name)
            result["breast_image"] = self._prepare_image_history(
                chest_image, "breast_image", self.breast_image_shape
            )
        return result

    def prepare_torch_obs(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        import torch

        numpy_obs = self.prepare_numpy_obs(obs)
        torch_obs = {}
        for key, value in numpy_obs.items():
            tensor = torch.from_numpy(value).unsqueeze(0)
            if self.device is not None:
                tensor = tensor.to(self.device)
            torch_obs[key] = tensor
        return torch_obs

    def metadata(self, policy_name: str, n_action_steps: int | None = None) -> Dict[str, Any]:
        return {
            "policy_name": policy_name,
            "action_horizon": int(n_action_steps) if n_action_steps is not None else None,
            "action_dim": self.action_dim,
            "n_obs_steps": self.n_obs_steps,
            "image_view_used": "both" if self.requires_chest_view else "head",
            "chest_view_accepted": self.requires_chest_view,
            "protocol": "legendvla_msgpack_numpy",
        }


class DiffusionPolicyServingEngine:
    def __init__(
        self,
        policy: Any,
        adapter: LegendVlaObservationAdapter,
        device: str | None = None,
        use_autocast: bool = False,
    ) -> None:
        self.policy = policy
        self.adapter = adapter
        self.device = device
        self.use_autocast = use_autocast

    def _autocast_context(self):
        if not self.use_autocast:
            return nullcontext()
        import torch

        device_type = torch.device(self.device or "cpu").type
        return torch.autocast(device_type=device_type)

    def infer(self, obs: Mapping[str, Any]) -> Dict[str, Any]:
        import torch

        torch_obs = self.adapter.prepare_torch_obs(obs)
        with self._autocast_context(), torch.inference_mode():
            result = self.policy.predict_action(torch_obs)
        action = result["action"][0].detach().to("cpu").float().numpy().astype(np.float32, copy=False)
        return {"pred_actions": action}

    def metadata(self) -> Dict[str, Any]:
        policy_name = type(self.policy).__name__
        n_action_steps = getattr(self.policy, "n_action_steps", None)
        return self.adapter.metadata(policy_name=policy_name, n_action_steps=n_action_steps)


def load_policy_checkpoint(
    ckpt_path: str | pathlib.Path,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
) -> tuple[Any, Any]:
    import dill
    import hydra
    import torch
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    ckpt_path = pathlib.Path(ckpt_path).expanduser()
    with ckpt_path.open("rb") as f:
        payload = torch.load(f, pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace_cls = hydra.utils.get_class(cfg._target_)
    workspace = workspace_cls(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.model
    if _get_nested(cfg, ("training", "use_ema"), False):
        policy = workspace.ema_model

    if num_inference_steps is not None:
        policy.num_inference_steps = int(num_inference_steps)
    if n_action_steps is not None:
        policy.n_action_steps = int(n_action_steps)

    policy.eval().to(torch.device(device))
    return policy, cfg


class WebsocketPolicyServer:
    def __init__(
        self,
        engine: DiffusionPolicyServingEngine,
        host: str,
        port: int,
        metadata: Mapping[str, Any] | None = None,
        log_obs_details: bool = False,
    ) -> None:
        self.engine = engine
        self.host = host
        self.port = int(port)
        self.metadata = dict(metadata or {})
        self.log_obs_details = log_obs_details

    @staticmethod
    def _format_obs_details(obs: Mapping[str, Any]) -> str:
        lines = [f"received observation fields: {list(obs.keys())}"]
        for key, value in obs.items():
            if isinstance(value, Mapping):
                lines.append(f"  {key}: dict keys={list(value.keys())}")
            elif hasattr(value, "shape"):
                lines.append(f"  {key}: shape={value.shape} dtype={getattr(value, 'dtype', None)}")
            else:
                lines.append(f"  {key}: type={type(value).__name__}")
        return "\n".join(lines)

    async def _handler(self, websocket: Any, *_args: Any) -> None:
        packer = msgpack_numpy.Packer()
        remote = getattr(websocket, "remote_address", None)
        logger.info("Connection from %s opened", remote)
        await websocket.send(packer.pack(self.metadata))

        prev_send_time = None
        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()

                recv_start = time.monotonic()
                raw_obs = await websocket.recv()
                recv_wait_time = time.monotonic() - recv_start
                obs = msgpack_numpy.unpackb(raw_obs)

                if self.log_obs_details:
                    logger.info("%s\n%s", remote, self._format_obs_details(obs))

                infer_start = time.monotonic()
                response = self.engine.infer(obs)
                infer_time = time.monotonic() - infer_start
                response["server_timing"] = {
                    "recv_wait_ms": recv_wait_time * 1000.0,
                    "infer_ms": infer_time * 1000.0,
                }
                if prev_send_time is not None:
                    response["server_timing"]["prev_send_ms"] = prev_send_time * 1000.0
                if prev_total_time is not None:
                    response["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                pack_start = time.monotonic()
                packed_response = packer.pack(response)
                response["server_timing"]["pack_ms"] = (time.monotonic() - pack_start) * 1000.0
                response["server_timing"]["pre_send_total_ms"] = (time.monotonic() - start_time) * 1000.0
                packed_response = packer.pack(response)

                send_start = time.monotonic()
                await websocket.send(packed_response)
                prev_send_time = time.monotonic() - send_start
                prev_total_time = time.monotonic() - start_time
            except Exception as exc:
                if _is_connection_closed(exc):
                    logger.info("Connection from %s closed", remote)
                    break
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    async def run(self) -> None:
        serve, process_request = _websocket_serve_helpers()
        async with serve(
            self._handler,
            self.host,
            self.port,
            compression=None,
            max_size=None,
            process_request=process_request,
        ) as server:
            await server.serve_forever()

    def serve_forever(self) -> None:
        asyncio.run(self.run())


def _is_connection_closed(exc: Exception) -> bool:
    try:
        import websockets

        return isinstance(exc, websockets.ConnectionClosed)
    except Exception:
        return False


def _websocket_serve_helpers():
    try:
        import websockets.asyncio.server as websocket_server

        def process_request(connection: Any, request: Any) -> Any:
            if getattr(request, "path", None) == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        return websocket_server.serve, process_request
    except Exception:
        import websockets

        def process_request(path: str, _headers: Any) -> Any:
            if path == "/healthz":
                return http.HTTPStatus.OK, [], b"OK\n"
            return None

        return websockets.serve, process_request


def build_engine_from_checkpoint(
    input_path: str | pathlib.Path,
    device: str,
    num_inference_steps: int | None = None,
    n_action_steps: int | None = None,
    use_autocast: bool = False,
) -> DiffusionPolicyServingEngine:
    policy, cfg = load_policy_checkpoint(
        input_path,
        device=device,
        num_inference_steps=num_inference_steps,
        n_action_steps=n_action_steps,
    )
    shape_meta = _to_plain_container(_get_nested(cfg, ("task", "shape_meta"), None) or cfg.shape_meta)
    n_obs_steps_cfg = _get_nested(cfg, ("n_obs_steps",), None)
    if n_obs_steps_cfg is None:
        n_obs_steps_cfg = getattr(policy, "n_obs_steps")
    adapter = LegendVlaObservationAdapter(shape_meta=shape_meta, n_obs_steps=int(n_obs_steps_cfg), device=device)
    return DiffusionPolicyServingEngine(policy=policy, adapter=adapter, device=device, use_autocast=use_autocast)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", required=True, help="Path to Diffusion Policy checkpoint")
    parser.add_argument("--host", default="0.0.0.0", help="WebSocket bind host")
    parser.add_argument("--port", type=int, default=8765, help="WebSocket bind port")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu")
    parser.add_argument("--num-inference-steps", type=int, default=None, help="Override diffusion inference steps")
    parser.add_argument("--n-action-steps", type=int, default=None, help="Override policy action chunk length")
    parser.add_argument("--log-obs-details", action="store_true", help="Log received observation fields and shapes")
    parser.add_argument("--autocast", action="store_true", help="Run policy inference under torch.autocast")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    engine = build_engine_from_checkpoint(
        args.input,
        device=args.device,
        num_inference_steps=args.num_inference_steps,
        n_action_steps=args.n_action_steps,
        use_autocast=args.autocast,
    )
    metadata = engine.metadata()
    logger.info("Serving %s on ws://%s:%d", metadata["policy_name"], args.host, args.port)
    WebsocketPolicyServer(
        engine=engine,
        host=args.host,
        port=args.port,
        metadata=metadata,
        log_obs_details=args.log_obs_details,
    ).serve_forever()


if __name__ == "__main__":
    main()
