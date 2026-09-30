import pytest
import torch
import torch.nn.functional as F
from ultralytics.utils.loss import KeypointLoss

from training.partial_pose import masked_keypoint_losses


def test_unknown_points_have_zero_coordinate_and_objectness_gradient():
    pred = torch.tensor([[[1.0, 1.0, 0.0], [2.0, 2.0, 0.5], [3.0, 3.0, -0.5]]], requires_grad=True)
    target = torch.tensor([[[0.0, 0.0, 2.0], [float("nan"), float("nan"), 0], [0, 0, 0]]])
    known = torch.tensor([[True, False, True]])
    coord, obj = masked_keypoint_losses(pred, target, known, torch.tensor([[100.0]]), torch.ones(3) / 3)
    (coord + obj).backward()
    assert torch.isfinite(coord + obj)
    assert pred.grad[0, 0, :2].abs().sum() > 0
    assert pred.grad[0, 1].abs().sum() == 0
    assert pred.grad[0, 2, :2].abs().sum() == 0
    assert pred.grad[0, 2, 2] > 0  # known absence still supervises negative objectness


def test_fully_labelled_targets_match_ultralytics_loss_and_gradients():
    torch.manual_seed(7)
    pred = torch.randn(2, 6, 3, requires_grad=True)
    target = torch.rand(2, 6, 3)
    target[..., 2] = torch.tensor([[2, 2, 0, 1, 2, 0], [0, 0, 0, 0, 0, 0]])
    known = torch.ones(2, 6, dtype=torch.bool)
    area = torch.tensor([[100.0], [200.0]])
    sigmas = torch.ones(6) / 6
    coord, obj = masked_keypoint_losses(pred, target, known, area, sigmas)
    visible = target[..., 2] != 0
    expected = KeypointLoss(sigmas)(pred, target, visible, area) + F.binary_cross_entropy_with_logits(pred[..., 2], visible.float())
    torch.testing.assert_close(coord + obj, expected)
    torch.testing.assert_close(torch.autograd.grad(coord + obj, pred, retain_graph=True)[0], torch.autograd.grad(expected, pred)[0])


@pytest.mark.parametrize("count", [0, 2])
def test_empty_or_all_unknown_batch_returns_differentiable_zero(count):
    pred = torch.zeros(count, 6, 3, requires_grad=True)
    target = torch.full_like(pred, float("nan"))
    coord, obj = masked_keypoint_losses(pred, target, torch.zeros(count, 6, dtype=torch.bool), torch.ones(count, 1), torch.ones(6) / 6)
    (coord + obj).backward()
    assert (coord + obj).item() == 0
    assert pred.grad.abs().sum() == 0


def test_mask_is_required_to_be_boolean():
    with pytest.raises(ValueError, match="boolean"):
        masked_keypoint_losses(torch.zeros(1, 6, 3), torch.zeros(1, 6, 3), torch.ones(1, 6), torch.ones(1, 1), torch.ones(6))
