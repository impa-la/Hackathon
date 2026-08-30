# /// <summary>
# Independent reproduction of the existing lightweight log-mel CNN reference
# /// </summary>

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


HeadCount = 5


def HertzToMel(Frequency: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + Frequency / 700.0)


def MelToHertz(MelValue: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, MelValue / 2595.0) - 1.0)


def BuildMelFilter(
    SampleRate: int,
    Nfft: int,
    MelBins: int,
    MinimumFrequency: float = 20.0,
) -> torch.Tensor:
    FrequencyBins = Nfft // 2 + 1
    MinimumMel = HertzToMel(torch.tensor(MinimumFrequency))
    MaximumMel = HertzToMel(torch.tensor(SampleRate / 2.0))
    MelPoints = torch.linspace(MinimumMel, MaximumMel, MelBins + 2)
    HertzPoints = MelToHertz(MelPoints)
    BinPoints = torch.floor((Nfft + 1) * HertzPoints / SampleRate).long()
    BinPoints = torch.clamp(BinPoints, 0, FrequencyBins - 1)
    Filter = torch.zeros(MelBins, FrequencyBins)
    for MelIndex in range(MelBins):
        Left = int(BinPoints[MelIndex])
        Center = max(int(BinPoints[MelIndex + 1]), Left + 1)
        Right = min(max(int(BinPoints[MelIndex + 2]), Center + 1), FrequencyBins)
        if Center > Left:
            Filter[MelIndex, Left:Center] = torch.linspace(0.0, 1.0, Center - Left)
        if Right > Center:
            Filter[MelIndex, Center:Right] = torch.linspace(1.0, 0.0, Right - Center)
    return Filter / Filter.sum(dim=1, keepdim=True).clamp_min(1e-8)


class LogMelExtractor(nn.Module):
    def __init__(self, SampleRate: int, Nfft: int, HopLength: int, MelBins: int) -> None:
        super().__init__()
        self.Nfft = Nfft
        self.HopLength = HopLength
        self.register_buffer("Window", torch.hann_window(Nfft), persistent=True)
        self.register_buffer(
            "MelFilter",
            BuildMelFilter(SampleRate, Nfft, MelBins),
            persistent=True,
        )

    def forward(
        self,
        Waveform: torch.Tensor,
        ValidSampleCounts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ValidSampleCounts is None:
            ValidSampleCounts = torch.full(
                (Waveform.shape[0],),
                Waveform.shape[-1],
                device=Waveform.device,
                dtype=torch.long,
            )
        ValidSampleCounts = ValidSampleCounts.to(Waveform.device, dtype=torch.long)
        SamplePositions = torch.arange(Waveform.shape[-1], device=Waveform.device)
        SampleMask = SamplePositions.unsqueeze(0) < ValidSampleCounts.unsqueeze(1)
        Waveform = Waveform * SampleMask.to(Waveform.dtype)
        Spectrum = torch.stft(
            Waveform,
            n_fft=self.Nfft,
            hop_length=self.HopLength,
            win_length=self.Nfft,
            window=self.Window,
            center=True,
            return_complex=True,
        )
        Power = Spectrum.abs().square()
        Features = torch.log1p(torch.einsum("mf,bft->bmt", self.MelFilter, Power))
        FramePositions = torch.arange(Features.shape[-1], device=Waveform.device)
        FrameCenters = FramePositions * self.HopLength
        FrameMask = FrameCenters.unsqueeze(0) < ValidSampleCounts.unsqueeze(1)
        ExpandedMask = FrameMask.unsqueeze(1).to(Features.dtype)
        ElementCount = (
            FrameMask.sum(dim=1, keepdim=True).to(Features.dtype)
            * Features.shape[1]
        ).clamp_min(2.0)
        Mean = (Features * ExpandedMask).sum(dim=(1, 2), keepdim=True)
        Mean = Mean / ElementCount.unsqueeze(-1)
        SquaredError = ((Features - Mean).square() * ExpandedMask).sum(
            dim=(1, 2), keepdim=True
        )
        StandardDeviation = (
            SquaredError / (ElementCount - 1.0).unsqueeze(-1)
        ).sqrt().clamp_min(1e-5)
        Normalized = (Features - Mean) / StandardDeviation
        Normalized = Normalized * ExpandedMask
        return Normalized.unsqueeze(1), FrameMask


class LogMelCnn(nn.Module):
    def __init__(self, Config: dict[str, Any]) -> None:
        super().__init__()
        BaseChannels = int(Config["base_channels"])
        self.Config = dict(Config)
        self.FeatureExtractor = LogMelExtractor(
            int(Config["sample_rate"]),
            int(Config["n_fft"]),
            int(Config["hop_length"]),
            int(Config["mel_bins"]),
        )
        self.Blocks = nn.ModuleList(
            (
                nn.Sequential(
                    nn.Conv2d(1, BaseChannels, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(BaseChannels),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(BaseChannels, BaseChannels * 2, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(BaseChannels * 2),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(BaseChannels * 2, BaseChannels * 4, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(BaseChannels * 4),
                    nn.SiLU(),
                ),
                nn.Sequential(
                    nn.Conv2d(BaseChannels * 4, BaseChannels * 4, kernel_size=3, padding=1),
                    nn.BatchNorm2d(BaseChannels * 4),
                    nn.SiLU(),
                ),
            )
        )
        self.Head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(BaseChannels * 4, HeadCount),
        )

    def forward(
        self,
        Waveform: torch.Tensor,
        ValidSampleCounts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # R4 is deliberately quality-first FP32. Reject any accidental return
        # to the numerically fragile CUDA AMP path before model compute begins.
        if Waveform.is_cuda and torch.is_autocast_enabled():
            raise RuntimeError("E01-R4 model compute forbids CUDA autocast")
        with torch.autocast(device_type=Waveform.device.type, enabled=False):
            Features, FrameMask = self.FeatureExtractor(
                Waveform.float(),
                ValidSampleCounts,
            )
        for BlockIndex, Block in enumerate(self.Blocks):
            Features = Block(Features)
            if BlockIndex < 3:
                FrameMask = F.max_pool1d(
                    FrameMask.unsqueeze(1).to(Features.dtype),
                    kernel_size=3 if BlockIndex else 5,
                    stride=2,
                    padding=1 if BlockIndex else 2,
                ).squeeze(1) > 0
            Features = Features * FrameMask[:, None, None, :].to(Features.dtype)
        PoolMask = FrameMask[:, None, None, :].to(Features.dtype)
        Denominator = (
            FrameMask.sum(dim=1).to(Features.dtype) * Features.shape[2]
        ).clamp_min(1.0)
        Pooled = (Features * PoolMask).sum(dim=(2, 3)) / Denominator.unsqueeze(1)
        return self.Head(Pooled)


def CountTrainableParameters(Model: nn.Module) -> int:
    return sum(Parameter.numel() for Parameter in Model.parameters() if Parameter.requires_grad)
