"""Small CPU regression checks for the training-time weighting rule."""

import torch
from torch import nn

from finetrainers.models.wan.physical_motion_phystrata import (
    _compute_motion_bmv_rel_l2,
    _motion_credit_weights,
    extract_predictor_weights,
)


def test_weight_shapes_bounds_and_budget():
    torch.manual_seed(42)
    target = torch.randn(4, 13, 16, 3, 3)
    video, interval = _motion_credit_weights(target, tau=2)
    assert video.shape == (4,)
    assert interval.shape == (4, 11)
    assert (video >= 0.6).all() and (video <= 1.4).all()
    assert (interval >= 0.6).all() and (interval <= 1.4).all()
    torch.testing.assert_close(interval.mean(dim=1), torch.ones(4), atol=1e-6, rtol=0)


def test_flat_video_and_zero_strength():
    target = torch.ones(2, 13, 16, 2, 2)
    video, interval = _motion_credit_weights(target, tau=2)
    torch.testing.assert_close(video, torch.ones(2))
    torch.testing.assert_close(interval, torch.ones(2, 11))
    video_off, interval_off = _motion_credit_weights(target, tau=2, beta_video=0, beta_frame=0)
    torch.testing.assert_close(video_off, torch.ones(2))
    torch.testing.assert_close(interval_off, torch.ones(2, 11))


def test_zero_strength_matches_original_drift():
    torch.manual_seed(7)
    target = torch.randn(2, 7, 16, 2, 2)
    predicted = target + 0.01 * torch.randn_like(target)
    video, interval = _motion_credit_weights(target, tau=2, beta_video=0, beta_frame=0)
    original = _compute_motion_bmv_rel_l2(predicted, target, tau=2)
    weighted = _compute_motion_bmv_rel_l2(
        predicted, target, tau=2, video_weights=video, interval_weights=interval,
    )
    torch.testing.assert_close(weighted, original)


def test_single_namespace_checkpoint_loading():
    predictor = nn.Linear(3, 2)
    state = {f"training_wrapper.{key}": value for key, value in predictor.state_dict().items()}
    extracted = extract_predictor_weights(state, predictor)
    assert set(extracted) == set(predictor.state_dict())
    for key in extracted:
        torch.testing.assert_close(extracted[key], predictor.state_dict()[key])


if __name__ == "__main__":
    test_weight_shapes_bounds_and_budget()
    test_flat_video_and_zero_strength()
    test_zero_strength_matches_original_drift()
    test_single_namespace_checkpoint_loading()
    print("motion-credit CPU checks passed")
