# type: ignore
import torch


# -----------------------------------------------------------------------------
# OrthgonalNesterov optimizer


class OrthogonalNesterov(torch.optim.Optimizer):
    """
    Some warnings: This optimizer assumes that all parameters passed in are 2D.
    It shouldn't be used for the embedding layer, the final fully connected layer, or {0,1}-D
    parameters; those should be optimized by a standard method (e.g., AdamW).
    To use it with 4D convolutional filters, it works well to flatten their last 3 dimensions.
    """

    def __init__(
        self,
        params,
        lr=4e-4,
        momentum=0.95,
        nesterov=True,
        zeropower_iters=5,
        eps=1e-5,
        betas=[0.9, 0.95],
        weight_decay=1e-5,
        vocab_dim=32768,  # hack for now
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            zeropower_iters=zeropower_iters,
            eps=eps,
            betas=betas,
            weight_decay=weight_decay,
            vocab_dim=vocab_dim,
        )
        super().__init__(params, defaults)

    @torch.no_grad
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            for p in group["params"]:
                state = self.state[p]
                g = p.grad
                if g is None:
                    continue
                if p.ndim == 2 and p.shape[0] != group["vocab_dim"] and p.shape[1] != group["vocab_dim"]:
                    # Newton-Schulz mode
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    g = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                    update = zeroth_power_via_newtonschulz5(g, steps=group["zeropower_iters"])
                    scale = update.numel() ** 0.5 / update.norm()
                    p.data.add_(update, alpha=-lr * 0.1 * scale)
                else:
                    # adam mode
                    if "step" not in state:
                        state["step"] = 0  # torch.tensor(0.0, dtype=torch.float32)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    # update step
                    state["step"] += 1
                    beta1, beta2 = group["betas"]
                    # wd
                    p.mul_(1 - lr * group["weight_decay"])

                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]

                    step_size = lr / bias_correction1

                    bias_correction2_sqrt = bias_correction2**0.5
                    denom = (state["exp_avg"].sqrt() / bias_correction2_sqrt).add_(group["eps"])
                    p.addcdiv_(state["exp_avg"], denom, value=-step_size)


@torch.compile
def zeroth_power_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. It turns out
    to be empirically effective to keep increasing the slope of the quintic at zero even beyond the
    point where it no longer converges to one everywhere after repeated application (so long as it
    stays relatively close to 1 across the interval). Our usage of a Newton-Schulz iteration as the
    orthogonalization method traces to Bernstein & Newhouse (2024) https://arxiv.org/abs/2409.20325
    who suggested its use for computing the preconditioners of Shampoo.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)






