import math
from typing import List

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding

    def forward(self, x):
        x_pad = F.pad(x, (self.__padding * 2, 0))
        return super().forward(x_pad)


class CausalTransposeConv1d(nn.ConvTranspose1d):
    def __init__(self, *args, padding: int = 0, output_padding: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.__padding = padding
        self.__output_padding = output_padding

    def forward(self, x):
        return super().forward(x)[..., : -(self.__padding * 2 - self.__output_padding)]


def WNCausalConv1d(*args, **kwargs):
    return weight_norm(CausalConv1d(*args, **kwargs))


def WNCausalTransposeConv1d(*args, **kwargs):
    return weight_norm(CausalTransposeConv1d(*args, **kwargs))


# 脚本化可提升模型速度约 1.4 倍
@torch.jit.script
def snake(x, alpha):
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x):
        return snake(x, self.alpha)


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class CausalResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1, kernel: int = 7, groups: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNCausalConv1d(
                dim,
                dim,
                kernel_size=kernel,
                dilation=dilation,
                padding=pad,
                groups=groups,
            ),
            Snake1d(dim),
            WNCausalConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        assert pad == 0
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class CausalEncoderBlock(nn.Module):
    def __init__(self, output_dim: int = 16, input_dim=None, stride: int = 1, groups=1):
        super().__init__()
        input_dim = input_dim or output_dim // 2
        self.block = nn.Sequential(
            CausalResidualUnit(input_dim, dilation=1, groups=groups),
            CausalResidualUnit(input_dim, dilation=3, groups=groups),
            CausalResidualUnit(input_dim, dilation=9, groups=groups),
            Snake1d(input_dim),
            WNCausalConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class CausalEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        latent_dim: int = 32,
        strides: list = [2, 4, 8, 8],
        depthwise: bool = False,
    ):
        super().__init__()
        # 创建第一层卷积
        self.block = [WNCausalConv1d(1, d_model, kernel_size=7, padding=3)]

        # 创建在下采样时通道数加倍的 EncoderBlock
        for stride in strides:
            d_model *= 2
            groups = d_model // 2 if depthwise else 1
            self.block += [CausalEncoderBlock(output_dim=d_model, stride=stride, groups=groups)]

        groups = d_model if depthwise else 1

        # 为均值 (mu) 和对数方差 (logvar) 创建两层卷积
        self.fc_mu = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)
        self.fc_logvar = WNCausalConv1d(d_model, latent_dim, kernel_size=3, padding=1)

        # 将 block 包装为 nn.Sequential
        self.block = nn.Sequential(*self.block)
        self.enc_dim = d_model

    def forward(self, x):
        hidden_state = self.block(x)
        return {
            "hidden_state": hidden_state,
            "mu": self.fc_mu(hidden_state),
            "logvar": self.fc_logvar(hidden_state),
        }


class NoiseBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = WNCausalConv1d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, T = x.shape
        noise = torch.randn((B, 1, T), device=x.device, dtype=x.dtype)
        h = self.linear(x)
        n = noise * h
        x = x + n
        return x


class CausalDecoderBlock(nn.Module):
    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        groups=1,
        use_noise_block: bool = False,
    ):
        super().__init__()
        layers = [
            Snake1d(input_dim),
            WNCausalTransposeConv1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
        ]
        if use_noise_block:
            layers.append(NoiseBlock(output_dim))
        layers.extend(
            [
                CausalResidualUnit(output_dim, dilation=1, groups=groups),
                CausalResidualUnit(output_dim, dilation=3, groups=groups),
                CausalResidualUnit(output_dim, dilation=9, groups=groups),
            ]
        )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransposeLastTwoDim(torch.nn.Module):
    def forward(self, x):
        return torch.transpose(x, -1, -2)


class CausalDecoder(nn.Module):
    def __init__(
        self,
        input_channel,
        channels,
        rates,
        depthwise: bool = False,
        d_out: int = 1,
        use_noise_block: bool = False,
    ):
        super().__init__()

        # 添加第一层卷积
        if depthwise:
            layers = [
                WNCausalConv1d(
                    input_channel,
                    input_channel,
                    kernel_size=7,
                    padding=3,
                    groups=input_channel,
                ),
                WNCausalConv1d(input_channel, channels, kernel_size=1),
            ]
        else:
            layers = [WNCausalConv1d(input_channel, channels, kernel_size=7, padding=3)]

        # 添加上采样 + 多尺度残差块 (MRF)
        for i, stride in enumerate(rates):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            groups = output_dim if depthwise else 1
            layers += [
                CausalDecoderBlock(
                    input_dim,
                    output_dim,
                    stride,
                    groups=groups,
                    use_noise_block=use_noise_block,
                )
            ]

        # 添加最终卷积层
        layers += [
            Snake1d(output_dim),
            WNCausalConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AudioVAE(nn.Module):
    """
    参数:
        encoder_dim: 编码器的维度 (默认 128)
        encoder_rates: 编码器的下采样率列表 (默认 [2, 5, 8, 8])
        latent_dim: 潜在空间维度 (默认 64)
        decoder_dim: 解码器的维度 (默认 1536)
        decoder_rates: 解码器的上采样率列表 (默认 [8, 8, 5, 2])
        depthwise: 是否使用深度可分离卷积 (默认 True)
        sample_rate: 音频采样率 (默认 16000)
        use_noise_block: 是否在解码器中使用噪声块 (默认 False)
    """

    def __init__(
        self,
        encoder_dim: int = 128,
        encoder_rates: List[int] = [2, 5, 8, 8],
        latent_dim: int = 64,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 5, 2],
        depthwise: bool = True,
        sample_rate: int = 16000,
        use_noise_block: bool = False,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.depthwise = depthwise

        self.use_noise_block = use_noise_block

        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))

        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = CausalEncoder(
            encoder_dim,
            latent_dim,
            encoder_rates,
            depthwise=depthwise,
        )

        self.decoder = CausalDecoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            depthwise=depthwise,
            use_noise_block=use_noise_block,
        )
        self.sample_rate = sample_rate
        self.chunk_size = math.prod(encoder_rates)

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate
        pad_to = self.hop_length
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))

        return audio_data

    def decode(self, z: torch.Tensor):
        """
        解码给定的潜在码并返回音频数据

        参数
        ----------
        z : Tensor[B x D x T]
            输入的连续量化表示

        返回
        -------
        Tensor[B x 1 x length]
            解码后的音频数据
        """
        return self.decoder(z)

    def encode(self, audio_data: torch.Tensor, sample_rate: int):
        """
        编码音频数据为潜在表示

        参数
        ----------
        audio_data : Tensor[B x 1 x T]
            输入的音频张量
        sample_rate : int
            音频采样率

        返回
        -------
        Tensor[B x D x T]
            编码得到的潜在表示 (mu)
        """
        if audio_data.ndim == 2:
            audio_data = audio_data.unsqueeze(1)

        audio_data = self.preprocess(audio_data, sample_rate)
        return self.encoder(audio_data)["mu"]
