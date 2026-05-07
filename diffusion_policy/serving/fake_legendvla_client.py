"""Fake LegendVLA-Inference client for probing the DP WebSocket server."""

from __future__ import annotations

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent.parent.parent))

import argparse
import asyncio
import time
from typing import Any, Dict

import numpy as np


DEFAULT_INTRINSICS_640X480 = np.array(
    [
        [388.0, 0.0, 320.0],
        [0.0, 388.0, 240.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _rot6d_identity() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)


def _make_rgb_history(horizon: int, height: int, width: int, camera: str) -> np.ndarray:
    """Return THWC uint8 RGB frames, matching cv_bridge rgb8 output in the ROS node."""
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    frames = []
    camera_bias = 31 if camera == "head" else 97
    for t in range(horizon):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        image[..., 0] = (x + camera_bias + t * 7).astype(np.uint8)
        image[..., 1] = (y + camera_bias // 2 + t * 5).astype(np.uint8)
        image[..., 2] = np.uint8((camera_bias + t * 17) % 256)
        frames.append(image)
    return np.stack(frames, axis=0)


def _make_state_history(horizon: int) -> np.ndarray:
    """Build the same 48D state layout used by model_interface_node.py."""
    states = []
    rot6d = _rot6d_identity()
    left_tips_base = np.array(
        [
            [0.02, 0.03, 0.04],
            [0.03, 0.04, 0.05],
            [0.04, 0.05, 0.06],
            [0.05, 0.06, 0.07],
            [0.06, 0.07, 0.08],
        ],
        dtype=np.float32,
    )
    right_tips_base = -left_tips_base
    for t in range(horizon):
        phase = float(t) / max(1, horizon - 1)
        left_wrist_pos = np.array([0.30 + 0.01 * phase, 0.12, 0.45], dtype=np.float32)
        right_wrist_pos = np.array([0.30 + 0.01 * phase, -0.12, 0.45], dtype=np.float32)
        left_tips = left_tips_base + np.array([0.001 * t, 0.0, 0.0], dtype=np.float32)
        right_tips = right_tips_base + np.array([-0.001 * t, 0.0, 0.0], dtype=np.float32)
        state = np.concatenate(
            [
                left_wrist_pos,
                right_wrist_pos,
                rot6d,
                rot6d,
                left_tips.reshape(-1),
                right_tips.reshape(-1),
            ],
            axis=0,
        )
        states.append(state)
    return np.stack(states, axis=0).astype(np.float32)


def _make_camera_extrinsics() -> Dict[str, np.ndarray]:
    head = np.eye(4, dtype=np.float64)
    chest = np.eye(4, dtype=np.float64)
    chest[:3, 3] = np.array([0.08, 0.0, -0.18], dtype=np.float64)
    return {"head": head, "chest": chest}


def build_fake_payload(
    image_horizon: int = 6,
    state_horizon: int = 18,
    height: int = 480,
    width: int = 640,
    instruction: str = "fake dual-camera smoke test",
    include_action_rtc: bool = False,
    action_rtc_len: int = 4,
) -> Dict[str, Any]:
    states = _make_state_history(state_horizon)
    action_rtc = None
    if include_action_rtc:
        action_rtc = np.zeros((action_rtc_len, states.shape[-1]), dtype=np.float32)

    return {
        "image": {
            "head": _make_rgb_history(image_horizon, height, width, "head"),
            "chest": _make_rgb_history(image_horizon, height, width, "chest"),
        },
        "camera_intrinsics": {
            "head": DEFAULT_INTRINSICS_640X480.copy(),
            "chest": DEFAULT_INTRINSICS_640X480.copy(),
        },
        "camera_extrinsics": _make_camera_extrinsics(),
        "instruction": instruction,
        "states": states,
        "action_rtc": action_rtc,
    }


async def run_client(args: argparse.Namespace) -> None:
    import websockets.asyncio.client as websocket_client
    from diffusion_policy.serving import msgpack_numpy

    uri = args.url or f"ws://{args.host}:{args.port}"
    packer = msgpack_numpy.Packer()
    async with websocket_client.connect(uri, compression=None, max_size=None) as websocket:
        metadata = msgpack_numpy.unpackb(await websocket.recv())
        print(f"server metadata: {metadata}")

        for request_idx in range(args.num_requests):
            payload = build_fake_payload(
                image_horizon=args.image_horizon,
                state_horizon=args.state_horizon,
                height=args.height,
                width=args.width,
                instruction=args.instruction,
                include_action_rtc=args.include_action_rtc,
                action_rtc_len=args.action_rtc_len,
            )
            start = time.monotonic()
            await websocket.send(packer.pack(payload))
            response = await websocket.recv()
            elapsed_ms = (time.monotonic() - start) * 1000.0
            if isinstance(response, str):
                raise RuntimeError(f"server returned error text:\n{response}")
            decoded = msgpack_numpy.unpackb(response)
            pred_actions = decoded["pred_actions"]
            timing = decoded.get("server_timing", {})
            print(
                f"[{request_idx}] pred_actions={pred_actions.shape} "
                f"dtype={pred_actions.dtype} rtt_ms={elapsed_ms:.2f} "
                f"infer_ms={timing.get('infer_ms', float('nan')):.2f}"
            )
            if pred_actions.shape[-1] != 48:
                raise RuntimeError(f"expected action dim 48, got {pred_actions.shape}")
            if args.sleep_ms > 0:
                await asyncio.sleep(args.sleep_ms / 1000.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=8765, help="Server port.")
    parser.add_argument("--url", default=None, help="Full ws:// URL. Overrides host/port.")
    parser.add_argument("--num-requests", type=int, default=3, help="Number of inference requests.")
    parser.add_argument("--sleep-ms", type=int, default=200, help="Delay between requests.")
    parser.add_argument("--width", type=int, default=640, help="RGB image width.")
    parser.add_argument("--height", type=int, default=480, help="RGB image height.")
    parser.add_argument("--image-horizon", type=int, default=6, help="THWC image history length.")
    parser.add_argument("--state-horizon", type=int, default=18, help="48D state history length.")
    parser.add_argument("--instruction", default="fake dual-camera smoke test", help="Instruction string.")
    parser.add_argument("--include-action-rtc", action="store_true", help="Include zero action_rtc chunk.")
    parser.add_argument("--action-rtc-len", type=int, default=4, help="action_rtc chunk length.")
    return parser.parse_args()


def main() -> None:
    asyncio.run(run_client(parse_args()))


if __name__ == "__main__":
    main()
