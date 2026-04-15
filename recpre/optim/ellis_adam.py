# type: ignore
import torch
from torch.optim import Optimizer
from torch import Tensor
from typing import List, Optional, Tuple, Union
from math import sqrt
import copy


def _parse_str_to_dtype(string_rep: str):
    if "bf16" in string_rep:
        return torch.bfloat16
    elif "f16" in string_rep or "fp16" in string_rep:
        return torch.float16
    else:
        return torch.float32


# an apple cobbler of many sources
class ELLISAdam(Optimizer):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-6,
        weight_decay: float = 1e-2,
        *,
        foreach: Optional[bool] = None,
        nesterov: bool = False,
        eps_adjustment: bool = False,
        update_clipping: bool = False,
        kahan_sum_compensation: bool = False,
        buffer_dtype: Optional[Union[torch.dtype, str]] = None,  # can be torch.float16 / torch.bfloat16
        running_init: bool = False,
        tensor_wise_finite_check: bool = False,
        tensor_wise_gradient_normalization: bool = False,
        adafactor_like_beta_corrections: bool = False,
        atan_adam: bool = False,
        decouple_wd: bool = True,
        brute_force_clip: Optional[float] = None,
        poly_ema_p: Optional[float] = None,
    ):
        defaults = dict(
            lr=torch.tensor(lr, dtype=torch.float32),
            init_lr=copy.deepcopy(lr),
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            nesterov=nesterov,
            eps_adjustment=eps_adjustment,
            update_clipping=update_clipping,
            kahan_sum_compensation=kahan_sum_compensation,
            buffer_dtype=_parse_str_to_dtype(buffer_dtype) if isinstance(buffer_dtype, str) else buffer_dtype,
            running_init=running_init,
            tensor_wise_finite_check=tensor_wise_finite_check,
            tensor_wise_gradient_normalization=tensor_wise_gradient_normalization,
            adafactor_like_beta_corrections=adafactor_like_beta_corrections,
            atan_adam=atan_adam,
            decouple_wd=decouple_wd,
            brute_force_clip=brute_force_clip,
            poly_ema_p=poly_ema_p,
        )
        self.arg_lr = lr
        if foreach:
            raise ValueError("Todo: reinstate a foreach version, minimizing additional mem alloc")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("foreach", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float32)

    @torch.no_grad()
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        kahan_comps,
        running_init: bool = False,
        buffer_dtype=None,
        kahan_sum_compensation: bool = False,
        tensor_wise_gradient_normalization: bool = False,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                _tensor_constructors = dict(memory_format=torch.preserve_format)
                if buffer_dtype is not None:
                    _tensor_constructors["dtype"] = buffer_dtype

                # note(crcrpar): Deliberately host `step` on CPU if both capturable and fused are off.
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0, dtype=torch.long)

                if kahan_sum_compensation:
                    state["kahan_comps"] = torch.zeros_like(p, **_tensor_constructors)
                else:
                    state["kahan_comps"] = None
                if running_init:
                    grad = p.grad if not tensor_wise_gradient_normalization else p.grad / p.grad.norm()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = grad.clone().to(**_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = grad.pow(2).clone().to(**_tensor_constructors)
                else:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, **_tensor_constructors)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, **_tensor_constructors)

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            state_steps.append(state["step"])
            kahan_comps.append(state["kahan_comps"])

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            kahan_comps = []
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                running_init=group["running_init"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
            )
            _single_tensor_modded_adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                kahan_comps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                init_lr=group["init_lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                nesterov=group["nesterov"],
                eps_adjustment=group["eps_adjustment"],
                update_clipping=group["update_clipping"],
                kahan_sum_compensation=group["kahan_sum_compensation"],
                buffer_dtype=group["buffer_dtype"],
                tensor_wise_finite_check=group["tensor_wise_finite_check"],
                tensor_wise_gradient_normalization=group["tensor_wise_gradient_normalization"],
                adafactor_like_beta_corrections=group["adafactor_like_beta_corrections"],
                atan_adam=group["atan_adam"],
                decouple_wd=group["decouple_wd"],
                brute_force_clip=group["brute_force_clip"],
                poly_ema_p=group["poly_ema_p"],
            )

        return loss


def _single_tensor_modded_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    kahan_comps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    init_lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    nesterov: bool = False,
    eps_adjustment: bool = False,
    update_clipping: bool = False,
    kahan_sum_compensation: bool = False,
    buffer_dtype=Optional[torch.dtype],
    tensor_wise_finite_check: bool = False,
    tensor_wise_gradient_normalization: bool = False,
    adafactor_like_beta_corrections: bool = False,
    atan_adam: bool = False,
    decouple_wd: bool = False,
    brute_force_clip: Optional[float] = None,
    poly_ema_p: Optional[float] = None,
):
    if adafactor_like_beta_corrections:
        # update group step
        step_t = state_steps[0]  # crime
        step_t += 1
        beta1 = (beta1**step_t - beta1) / (beta1**step_t - 1)
        beta2 = (beta2**step_t - beta2) / (beta2**step_t - 1)

    if poly_ema_p is not None:
        step_t = state_steps[0]  # crime
        # beta1 = step_t / (step_t + poly_ema_p)
        beta2 = step_t / (step_t + poly_ema_p)  # palm: 1 - step_t ** -0.8

    if nesterov:
        alpha = 2 * (1 - beta1) - (1 - beta1) ** 2  # only for nesterov to fuse the two lerps

    for i, param in enumerate(params):
        grad = grads[i].to(buffer_dtype)
        if tensor_wise_finite_check and (~torch.isfinite(grad)).sum() > 0:
            continue

        if tensor_wise_gradient_normalization:
            grad = grad / grad.norm()
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]
        kahan_comp = kahan_comps[i]

        # Decay the first and second moment running average coefficient
        if nesterov:
            # Only difference between NAdamW and AdamW in this implementation.
            # The official PyTorch implementation of NAdam uses a different algorithm.
            # We undo these ops later on, which could cause numerical issues but saves
            # us from having to make an extra copy of the gradients.
            exp_avg.lerp_(grad, alpha)
        else:
            exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        step_size = lr.clone() if isinstance(lr, torch.Tensor) else lr

        if update_clipping:
            rms = grad.pow(2).div_(exp_avg_sq.clamp_(min=eps**2)).mean().sqrt()  # impl like optimi
            step_size = step_size / rms.clamp(min=1.0)

        if not adafactor_like_beta_corrections:
            step_t += 1
            bias_correction1 = 1 - beta1**step_t
            bias_correction2 = 1 - beta2**step_t
            bias_correction2_sqrt = sqrt(bias_correction2)

            step_size = step_size / bias_correction1
        else:
            bias_correction2 = 1.0
            bias_correction2_sqrt = 1.0

        # Actual adam step
        if kahan_sum_compensation:
            # Perform stepweight decay
            if decouple_wd:
                kahan_comp.mul_(1 - lr / init_lr * weight_decay)
            else:
                kahan_comp.mul_(1 - lr * weight_decay)
            if atan_adam:
                # a = b = 1
                kahan_comp.add_(torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt)), alpha=-step_size)
            elif eps_adjustment:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                kahan_comp.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)
            # update weights with kahan compensation using grad as temp buffer
            grad.copy_(param.detach())
            param.add_(kahan_comp)
            # save error back to kahan compensation for next iteration
            kahan_comp.add_(grad.sub_(param))
        else:
            # Perform stepweight decay
            if decouple_wd:
                param.mul_(1 - lr / init_lr * weight_decay)
            else:
                param.mul_(1 - lr * weight_decay)
            if atan_adam:
                update = torch.atan2(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt))
                if brute_force_clip is not None:
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.add_(update, alpha=-step_size)
            elif eps_adjustment:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt()
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.div(bias_correction2).add_(eps**2).sqrt(), value=-step_size)
            else:
                if brute_force_clip is not None:
                    update = exp_avg / exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)
                    param.add_(update / torch.clamp(update.norm(), min=brute_force_clip), alpha=-step_size)
                else:
                    param.addcdiv_(exp_avg, exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps), value=-step_size)

        # undo nadam
        if nesterov:
            exp_avg.lerp_(grad, 1 - 1 / beta1)






