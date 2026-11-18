"""Receiver-domain losses and vector–Jacobian products."""
import numpy as np
import torch

from .soft_dtw import PreparedSoftDTWDivergence


class SoftDTW:
    def __init__(self):
        self.observed = {}

    def value_gradient(self, predicted, observed, dt, scale, indices, shot, gradient):
        # Retain the MPI trace order used for the reference shot reduction.
        value = 0.0
        cotangent = np.zeros_like(predicted) if gradient else None
        count = predicted.shape[1] * predicted.shape[2]
        for stations in indices:
            if not len(stations):
                continue
            prediction = torch.tensor(
                predicted[:, stations, :], dtype=torch.float64, requires_grad=gradient
            )
            target = (
                observed[:, stations, :].transpose(1, 2, 0).reshape(-1, len(observed))
            )
            with torch.set_grad_enabled(gradient):
                traces = prediction.permute(1, 2, 0).reshape(-1, len(predicted))
                for start in range(0, len(traces), 32):
                    key = (shot, tuple(stations), start)
                    if key not in self.observed:
                        self.observed[key] = PreparedSoftDTWDivergence(
                            torch.as_tensor(target[start : start + 32]),
                            dt,
                            scale=scale,
                            gamma=0.1,
                            decimation=5,
                            anti_alias_cutoff_hz=8.0,
                            anti_alias_transition_hz=2.0,
                        )
                    term = (
                        self.observed[key]
                        .trace_losses(traces[start : start + 32])
                        .sum()
                        / count
                    )
                    value += term.item()
                    if gradient:
                        (derivative,) = torch.autograd.grad(term, prediction)
                        cotangent[:, stations, :] += derivative.numpy()
        return value, cotangent


class TensorMisfit:
    """A scalar PyTorch loss over a complete time × receiver × component shot."""

    def __init__(self, tensor_loss):
        self.tensor_loss = tensor_loss

    def value_gradient(self, predicted, observed, dt, scale, indices, shot, gradient):
        with torch.set_grad_enabled(gradient):
            prediction = torch.tensor(
                predicted, dtype=torch.float64, requires_grad=gradient
            )
            loss = self.tensor_loss(prediction, torch.from_numpy(observed), dt, scale)
            if loss.ndim or loss.dtype != torch.float64:
                raise ValueError("The misfit must return one float64 scalar per shot")
            cotangent = (
                torch.autograd.grad(loss, prediction)[0].numpy() if gradient else None
            )
        return loss.item(), cotangent
