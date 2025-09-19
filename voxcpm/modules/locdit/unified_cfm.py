import torch
from typing import List
from voxcpm.modules.locdit.local_dit import VoxCPMLocDiT
import math
from pydantic import BaseModel


class CfmConfig(BaseModel):
    sigma_min: float = 1e-06
    solver: str = "euler"
    t_scheduler: str = "log-norm"


class UnifiedCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        cfm_params: CfmConfig,
        estimator: VoxCPMLocDiT,
        mean_mode: bool = False,
    ):
        super().__init__()
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.t_scheduler = cfm_params.t_scheduler
        self.in_channels = in_channels
        self.mean_mode = mean_mode

        # 仅在此更改估计器的架构
        self.estimator = estimator

    @torch.inference_mode()
    def forward(
        self,
        mu: torch.Tensor,
        n_timesteps: int,
        patch_size: int,
        cond: torch.Tensor,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        sway_sampling_coef: float = 1.0, 
        use_cfg_zero_star: bool = True,
    ):
        """前向扩散

        参数:
            mu (torch.Tensor): 编码器的输出
                shape: (batch_size, n_feats)
            n_timesteps (int): 扩散步数
            cond: 未使用，保留以备将来使用
            temperature (float, optional): 用于缩放噪声的温度。默认值为 1.0。

        返回:
            sample: 生成的 mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        b, c = mu.shape
        t = patch_size
        z = torch.randn((b, self.in_channels, t), device=mu.device, dtype=mu.dtype) * temperature

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        # 摆动采样策略
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        return self.solve_euler(z, t_span=t_span, mu=mu, cond=cond, cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)

    def optimized_scale(self, positive_flat, negative_flat):
        dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
        squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
        
        st_star = dot_product / squared_norm
        return st_star

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        cond: torch.Tensor,
        cfg_value: float = 1.0,
        use_cfg_zero_star: bool = True,
    ):
        """
        固定的欧拉求解器用于 ODE。
        参数:
            x (torch.Tensor): 随机噪声
            t_span (torch.Tensor): n_timesteps 插值
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): 编码器的输出
                shape: (batch_size, n_feats)
            cond: 条件 —— 前缀提示
            cfg_value (float, optional): 引导的 cfg 值。默认值为 1.0。
        """
        t, _, dt = t_span[0], t_span[-1], t_span[0] - t_span[1]

        sol = []
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = 0.
            else:
                # 在 VoiceBox 中引入的 Classifier-Free Guidance 推理
                b = x.size(0)
                x_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                mu_in = torch.zeros([2 * b, mu.size(1)], device=x.device, dtype=x.dtype)
                t_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                dt_in = torch.zeros([2 * b], device=x.device, dtype=x.dtype)
                cond_in = torch.zeros([2 * b, self.in_channels, x.size(2)], device=x.device, dtype=x.dtype)
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                # 目前未使用
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                dphi_dt = self.estimator(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
                
                if use_cfg_zero_star:
                    positive_flat = dphi_dt.view(b, -1)
                    negative_flat = cfg_dphi_dt.view(b, -1)
                    st_star = self.optimized_scale(positive_flat, negative_flat)
                    st_star = st_star.view(b, *([1] * (len(dphi_dt.shape) - 1)))
                else:
                    st_star = 1.0
                
                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return sol[-1]
