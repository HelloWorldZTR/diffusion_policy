import os
import sys

import numpy as np
import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

torch = pytest.importorskip("torch")

from diffusion_policy.serving.legendvla_policy_server import (  # noqa: E402
    DiffusionPolicyServingEngine,
    LegendVlaObservationAdapter,
)
from diffusion_policy.serving.fake_legendvla_client import build_fake_payload  # noqa: E402


def _shape_meta(image_size=4):
    return {
        "obs": {
            "image": {"shape": [3, image_size, image_size], "type": "rgb"},
            "breast_image": {"shape": [3, image_size, image_size], "type": "rgb"},
            "state": {"shape": [48], "type": "low_dim"},
        },
        "action": {"shape": [48]},
    }


def _payload(t=2, image_size=4, head_value=10, chest_value=20):
    head = np.full((t, image_size, image_size, 3), head_value, dtype=np.uint8)
    chest = np.full((t, image_size, image_size, 3), chest_value, dtype=np.uint8)
    states = np.arange(t * 48, dtype=np.float32).reshape(t, 48)
    return {"image": {"head": head, "chest": chest}, "states": states}


def test_dual_view_payload_maps_head_and_chest_to_policy_obs():
    adapter = LegendVlaObservationAdapter(_shape_meta(), n_obs_steps=2, device="cpu")
    obs = adapter.prepare_numpy_obs(_payload())

    assert set(obs.keys()) == {"image", "breast_image", "state"}
    assert obs["image"].shape == (2, 3, 4, 4)
    assert obs["breast_image"].shape == (2, 3, 4, 4)
    np.testing.assert_allclose(obs["image"], 10.0 / 255.0)
    np.testing.assert_allclose(obs["breast_image"], 20.0 / 255.0)


def test_single_view_legacy_payload_is_rejected_for_dual_image_policy():
    adapter = LegendVlaObservationAdapter(_shape_meta(), n_obs_steps=2, device="cpu")
    payload = {"image": np.zeros((2, 4, 4, 3), dtype=np.uint8), "states": np.zeros((2, 48), dtype=np.float32)}

    with pytest.raises(ValueError, match="camera dict"):
        adapter.prepare_numpy_obs(payload)


def test_uint8_thwc_images_become_batched_torch_tchw_float():
    adapter = LegendVlaObservationAdapter(_shape_meta(), n_obs_steps=2, device="cpu")
    obs = adapter.prepare_torch_obs(_payload())

    assert obs["image"].shape == (1, 2, 3, 4, 4)
    assert obs["breast_image"].shape == (1, 2, 3, 4, 4)
    assert obs["image"].dtype == torch.float32
    assert float(obs["image"].min()) >= 0.0
    assert float(obs["image"].max()) <= 1.0


def test_short_histories_repeat_left_pad_and_long_histories_keep_latest():
    adapter = LegendVlaObservationAdapter(_shape_meta(), n_obs_steps=3, device="cpu")

    short = _payload(t=1, head_value=7, chest_value=9)
    short_obs = adapter.prepare_numpy_obs(short)
    assert short_obs["state"].shape == (3, 48)
    np.testing.assert_allclose(short_obs["state"][0], short["states"][0])
    np.testing.assert_allclose(short_obs["state"][1], short["states"][0])
    np.testing.assert_allclose(short_obs["state"][2], short["states"][0])

    long = _payload(t=5)
    long_obs = adapter.prepare_numpy_obs(long)
    np.testing.assert_allclose(long_obs["state"], long["states"][-3:])


class _FakePolicy:
    n_action_steps = 8

    def __init__(self):
        self.last_obs = None

    def predict_action(self, obs):
        self.last_obs = obs
        action = torch.arange(8 * 48, dtype=torch.float32).reshape(1, 8, 48)
        return {"action": action}


def test_engine_response_uses_pred_actions_contract():
    policy = _FakePolicy()
    adapter = LegendVlaObservationAdapter(_shape_meta(), n_obs_steps=2, device="cpu")
    engine = DiffusionPolicyServingEngine(policy=policy, adapter=adapter, device="cpu")

    response = engine.infer(_payload())

    assert set(response.keys()) == {"pred_actions"}
    assert response["pred_actions"].shape == (8, 48)
    assert response["pred_actions"].dtype == np.float32
    assert set(policy.last_obs.keys()) == {"image", "breast_image", "state"}


def test_fake_client_payload_matches_model_interface_format():
    payload = build_fake_payload(image_horizon=6, state_horizon=18, height=480, width=640)

    assert set(payload.keys()) == {
        "image",
        "camera_intrinsics",
        "camera_extrinsics",
        "instruction",
        "states",
        "action_rtc",
    }
    assert set(payload["image"].keys()) == {"head", "chest"}
    assert payload["image"]["head"].shape == (6, 480, 640, 3)
    assert payload["image"]["chest"].shape == (6, 480, 640, 3)
    assert payload["image"]["head"].dtype == np.uint8
    assert payload["camera_intrinsics"]["head"].shape == (3, 3)
    assert payload["camera_extrinsics"]["chest"].shape == (4, 4)
    assert payload["states"].shape == (18, 48)
    assert payload["states"].dtype == np.float32
    assert payload["action_rtc"] is None
