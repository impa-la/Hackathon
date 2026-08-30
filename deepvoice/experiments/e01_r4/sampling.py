# /// <summary>
# Deterministic group-first balanced E01 sampling contract
# /// </summary>

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from .records import AudioRecord


Strata = ("speech_real", "speech_fake", "music_real", "music_fake")


def ClassifyTrainingStratum(Record: AudioRecord) -> str | None:
    if Record.Dataset == "ljspeech-1.1":
        return "speech_real"
    if Record.Dataset == "wavefake-1.2.0":
        if Record.SourceFamily in ("common_voice_prompt", "jsut_basic5000"):
            return None
        return "speech_fake"
    if Record.Dataset == "fma-small":
        return "music_real"
    if Record.Dataset == "aime-open-model-subset":
        return "music_fake"
    raise ValueError(f"Unsupported training dataset: {Record.Dataset}")


def BuildGroupIndex(
    Records: Sequence[AudioRecord],
) -> dict[str, dict[str, list[int]]]:
    GroupIndex: dict[str, dict[str, list[int]]] = {
        Stratum: defaultdict(list) for Stratum in Strata
    }
    for RecordIndex, Record in enumerate(Records):
        Stratum = ClassifyTrainingStratum(Record)
        if Stratum is None:
            continue
        GroupIndex[Stratum][Record.ContentGroupKey].append(RecordIndex)
    for Stratum in Strata:
        if not GroupIndex[Stratum]:
            raise ValueError(f"Training stratum has no groups: {Stratum}")
    return GroupIndex


def BuildMusicFakeProviderGroups(
    Records: Sequence[AudioRecord],
    GroupIndex: dict[str, dict[str, list[int]]],
) -> dict[str, dict[str, list[int]]]:
    ProviderGroups: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for GroupKey, Indices in GroupIndex["music_fake"].items():
        for Index in Indices:
            Provider = Records[Index].GeneratorOrProvider
            ProviderGroups[Provider][GroupKey].append(Index)
    return {
        Provider: {
            GroupKey: GroupIndices
            for GroupKey, GroupIndices in sorted(Groups.items())
        }
        for Provider, Groups in sorted(ProviderGroups.items())
    }


class GroupFirstBalancedSampler(Sampler[int]):
    def __init__(
        self,
        Records: Sequence[AudioRecord],
        SamplesPerEpoch: int,
        Seed: int,
    ) -> None:
        if SamplesPerEpoch <= 0 or SamplesPerEpoch % len(Strata) != 0:
            raise ValueError("Samples per epoch must be a positive multiple of four")
        self.GroupIndex = BuildGroupIndex(Records)
        self.Records = list(Records)
        self.SpeechPairGroups = sorted(
            set(self.GroupIndex["speech_real"])
            & set(self.GroupIndex["speech_fake"])
        )
        if not self.SpeechPairGroups:
            raise ValueError("Speech real/fake strata have no paired content groups")
        self.MusicFakeProviderGroups = BuildMusicFakeProviderGroups(
            self.Records,
            self.GroupIndex,
        )
        self.SamplesPerEpoch = SamplesPerEpoch
        self.Seed = Seed
        self.Epoch = 0

    def SetEpoch(self, Epoch: int) -> None:
        if Epoch < 0:
            raise ValueError("Epoch must be nonnegative")
        self.Epoch = Epoch

    def __len__(self) -> int:
        return self.SamplesPerEpoch

    def __iter__(self) -> Iterator[int]:
        Generator = random.Random(self.Seed + self.Epoch * 1_000_003)
        MusicRealGroups = sorted(self.GroupIndex["music_real"])
        MusicFakeProviders = sorted(self.MusicFakeProviderGroups)
        QuadCount = self.SamplesPerEpoch // len(Strata)
        OutputIndices = []
        for _ in range(QuadCount):
            SpeechGroup = Generator.choice(self.SpeechPairGroups)
            MusicRealGroup = Generator.choice(MusicRealGroups)
            MusicFakeProvider = Generator.choice(MusicFakeProviders)
            MusicFakeGroup = Generator.choice(
                sorted(self.MusicFakeProviderGroups[MusicFakeProvider])
            )
            Quad = [
                Generator.choice(self.GroupIndex["speech_real"][SpeechGroup]),
                Generator.choice(self.GroupIndex["speech_fake"][SpeechGroup]),
                Generator.choice(self.GroupIndex["music_real"][MusicRealGroup]),
                Generator.choice(
                    self.MusicFakeProviderGroups[MusicFakeProvider][MusicFakeGroup]
                ),
            ]
            Generator.shuffle(Quad)
            OutputIndices.extend(Quad)
        return iter(OutputIndices)


def SummarizeSampler(Records: Sequence[AudioRecord]) -> list[dict[str, int | str]]:
    GroupIndex = BuildGroupIndex(Records)
    Rows = []
    for Stratum in Strata:
        Rows.append(
            {
                "scope": "train_no_test",
                "stratum": Stratum,
                "group_count": len(GroupIndex[Stratum]),
                "row_count": sum(len(Indices) for Indices in GroupIndex[Stratum].values()),
                "target_fraction": 0.25,
            }
        )
    return Rows
