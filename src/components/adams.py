import math

import torch as th
from torch.optim import Optimizer


class AdamS(Optimizer):
    """Adam + Stable Weight Decay (SWD)，Xie et al. 2023 "On the Overlooked
    Pitfalls of Weight Decay and How to Mitigate Them: A Gradient-Norm
    Perspective"（参考实现 zeke-xie/stable-weight-decay-regularization 的
    swd_optim/adams.py）。

    跟 AdamW 的唯一区别是 decay 项的尺度：AdamW 每步把参数乘 (1 - lr*wd)，
    而梯度那一项是 lr * m_hat / sqrt(v_hat)——两者不在一个尺度上，v_hat 很小
    的时候（RL 里 actor 梯度经常是 1e-3 量级）梯度步远大于 decay 步，decay
    基本形同虚设，权重照样涨。SWD 把 decay 项也除以 sqrt(v_bar)（v_bar 是
    v_hat 在所有参数上的均值，一个标量），让 decay 跟自适应梯度步同尺度：

        theta <- theta * (1 - lr * wd / sqrt(v_bar))
        theta <- theta - lr / bc1 * m / (sqrt(v / bc2) + eps)

    v_bar 是全局标量而不是逐元素的 v_hat，所以 decay 对所有参数是同一个
    收缩比例（逐元素除会变成 Adam 的 L2 正则那种被 v 抵消的形式，恰好是
    要避免的）。wd 的量级因此跟 SGD 的 weight decay 可比（论文建议
    5e-4 量级），比 AdamW 通常用的值要"有效"得多——同一个 wd 数值在这里的
    收缩力约是 AdamW 的 1/sqrt(v_bar) 倍。

    只保留 FPO++ learner 用到的路径：单一 lr/betas/eps/weight_decay，无
    amsgrad，无 foreach/fused。
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @th.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with th.enable_grad():
                loss = closure()

        # 第一遍：更新一阶/二阶矩，同时累积 v_hat 的全局和（跨所有 param
        # group、所有参数），用来算 SWD 的标量尺度 sqrt(v_bar)。
        param_size = 0
        exp_avg_sq_hat_sum = 0.0
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamS does not support sparse gradients")
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = th.zeros_like(p, memory_format=th.preserve_format)
                    state["exp_avg_sq"] = th.zeros_like(p, memory_format=th.preserve_format)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction2 = 1 - beta2 ** state["step"]
                exp_avg_sq_hat_sum += exp_avg_sq.sum().item() / bias_correction2
                param_size += p.numel()

        if param_size == 0:
            return loss
        exp_avg_sq_hat_mean = exp_avg_sq_hat_sum / param_size
        # 记下来供诊断（learner 可以直接读 opt.last_v_bar_sqrt 看 decay 的实
        # 际尺度）。加个下界防止 v_bar 恰好为 0（比如第一步全零梯度）时除零。
        self.last_v_bar_sqrt = max(math.sqrt(exp_avg_sq_hat_mean), 1e-16)

        # 第二遍：先做 stable weight decay，再做 Adam 步。
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                if wd != 0:
                    p.mul_(1 - group["lr"] * wd / self.last_v_bar_sqrt)
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group["eps"])
                step_size = group["lr"] / bias_correction1
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
