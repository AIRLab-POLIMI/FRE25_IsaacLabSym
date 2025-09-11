import torch


class UniversalController:
    def __init__(self, nEnvs: int, device: torch.device) -> None:
        self.nEnvs = nEnvs
        self.device = device

        self.vs = torch.zeros((nEnvs, 1), device=device)
        self.phis = torch.zeros((nEnvs, 1), device=device)
        self.ws = torch.zeros((nEnvs, 1), device=device)

    def reset(self, env_ids: torch.Tensor):
        self.vs[env_ids] = 0.0
        self.phis[env_ids] = 0.0
        self.ws[env_ids] = 0.0

    
