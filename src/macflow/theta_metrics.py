"""Read-only parameter trajectory metrics, measured after each train() call."""

import torch as th


class ThetaMetrics:
    def __init__(self):
        self.reference = None
        self.previous = None
        self.previous_step = None

    @th.no_grad()
    def measure(self, parameters):
        flat = th.cat([p.detach().reshape(-1) for p in parameters])
        if self.reference is None:
            # Match FPO: the first post-train snapshot becomes the fixed ref.
            self.reference = flat.clone()
            self.previous = flat.clone()
        else:
            self.reference = self.reference.to(flat.device)
            self.previous = self.previous.to(flat.device)
            if self.previous_step is not None:
                self.previous_step = self.previous_step.to(flat.device)

        step = flat - self.previous
        step_norm = step.norm()
        cosine = flat.new_zeros(())
        valid = False
        if self.previous_step is not None:
            previous_norm = self.previous_step.norm()
            if step_norm > 0 and previous_norm > 0:
                cosine = (th.dot(step, self.previous_step) / (step_norm * previous_norm)).clamp(-1, 1)
                valid = True
        metrics = {
            "theta_step_norm": step_norm.item(),
            "theta_step_cos": cosine.item(),
            "theta_disp_norm": (flat - self.reference).norm().item(),
            "theta_step_cos_valid": float(valid),
        }
        self.previous = flat.clone()
        self.previous_step = step.clone()
        return metrics

    def state_dict(self):
        return {name: None if value is None else value.detach().cpu().clone()
                for name in ("reference", "previous", "previous_step")
                for value in (getattr(self, name),)}

    def load_state_dict(self, state):
        for name in ("reference", "previous", "previous_step"):
            value = state[name]
            setattr(self, name, None if value is None else value.detach().clone())
