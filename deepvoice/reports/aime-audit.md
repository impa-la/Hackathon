DATA_READINESS: BLOCKED

# AIME pinned-revision 데이터 감사

감사 대상은 `disco-eth/AIME` revision `b84d4be5eda830b6eb714998569dba73530f2601`이다. 원본은 `C:\Users\MY PC\Desktop\Hackathon\deepvoice\data\raw\aime\b84d4be5eda830b6eb714998569dba73530f2601`에서 읽기 전용으로 검사했으며, 검사 전후 README와 36개 parquet의 크기·mtime이 모두 같았다.

## 판정

- **AIME 선택 subset 기술 판정: PASS.** 36/36 shard의 크기, 공식 LFS SHA-256, 행 수, 순수-provider 조건이 모두 일치했고 1,116/1,116 오디오가 전부 decode됐다.
- **AIME 선택 subset 학습 권리 판정: GO_WITH_ATTRIBUTION.** AIME 제작자가 자신들이 생성한 6,000곡을 CC BY 4.0으로 직접 배포했고, 사용자는 CC-BY/NC/SA 비영리 자원을 허용했다. 선택된 9개 provider에서는 생성 output의 재라이선스 또는 비경쟁 detector 학습을 명시적으로 금지하는 조항을 찾지 못했다. 따라서 이 1,116행은 AIME attribution을 유지해 학습할 수 있다. exact generation revision·service tier 공백은 학습 차단이 아니라 provenance-quality 경고로 남긴다.
- **Suno v3/v3.5와 Udio: EXCLUDED_HOLD.** 이번 36개 shard에는 포함하지 않았다. AIME 논문과 README에는 생성 당시 계정 tier와 당시 약관이 없다. 현재 Suno 약관도 유료 tier와 무료/Basic tier의 권리를 다르게 정하고 다른 AI 모델 학습 용도를 제한하므로, 단순 CC BY 표기만으로 포함하지 않는다.
- **전체 저장소 콘텐츠 감사: 미완료.** 메타데이터상 전체는 210 shards, 6,500 rows, 62,281,475,287 bytes지만 실제 오디오 전수 검사는 36 shards, 1,116 rows, 1,246,815,838 bytes에 한정된다.
- **독립 학습셋 판정: 부적합.** 선택 subset 1,116개는 모두 생성음이어서 real/fake 이진 분류를 단독으로 학습하거나 평가할 수 없다. real 500개인 MTG-Jamendo는 이번 subset 밖이며 per-track license 필터가 필요하다.
- **DeepVoice 종합 판정: BLOCKED.** AIME curated subset은 사용할 수 있지만 별도 필수 축인 WaveFake 취득·전수 감사가 끝나지 않았다. AIME 6,500행 전체를 추가 사용할 경우에는 미다운로드 174 shards 감사가 별도로 필요하다.

현재 허용되는 행동은 `aime-curated-manifest.csv`에서 `training_eligible=true`인 1,116행을 attribution과 함께 비영리 대회 학습·평가에 투입하는 것이다. Suno, Udio, MTG-Jamendo와 미다운로드 shard를 이 판단에 합쳐서는 안 된다.

## 출처와 범위

| 항목 | 전체 저장소 projection | 실제 관측 subset |
|---|---:|---:|
| parquet shards | 210 | 36 |
| rows | 6,500 | 1,116 |
| bytes | 62,281,475,287 | 1,246,815,838 |
| categories/providers | 13 × 500 | 9 × 124 |
| 오디오 content decode | 미수행 | 전수 1,116 |

전체 projection은 parquet에서 `id`, `model`만 읽어 산출했으며 오류는 0이었다. 실제 subset은 요청한 open-model 후보 9종마다 provider가 섞이지 않은 shard 4개를 ID 범위 전체에 균등하게 고른 것이다. 선택 계획과 실제 다운로드 manifest는 각각 `aime-subset-acquisition-plan.csv`, `aime-subset-download-manifest.csv`에 있다. README SHA-256은 `8adaaeb78a4a04f49e0b14d3b6a791f1dfb257972cd3f1e02740023710bf7880`이다.

[고정 revision README](https://huggingface.co/datasets/disco-eth/AIME/blob/b84d4be5eda830b6eb714998569dba73530f2601/README.md)는 `id`, `model`, `description`, `audio` schema, 6,000 generated + 500 MTG-Jamendo 구성, 생성 오디오 CC BY 4.0, MTG의 per-track license, description의 CC BY-NC-SA 4.0을 명시한다. [공식 논문](https://arxiv.org/abs/2506.19085)은 12개 생성 모델에서 각 500곡, 총 6,000곡을 만들고 공개했다고 설명한다.

## schema·타깃·결측

실제 schema는 `id: String`, `model: String`, `description: String`, `audio: Struct(path: String, bytes: Binary)`이며 네 필드의 결측은 모두 0이다. ID는 1,116개 모두 고유하고 provider별 124개다. `model`은 실제 생성기 provenance label이다. 이 subset에서는 모두 synthetic이므로 `is_fake=1`은 파생 가능하지만, real class가 없어서 이 subset 단독의 분류 타깃으로는 불완전하다.

## 오디오 전수 감사

FFprobe로 container/stream을 읽고 FFmpeg로 모든 파일을 interleaved f32le PCM으로 끝까지 decode했다. 무음은 `abs(sample) < 1e-4`, clipping은 `abs(sample) >= 0.999`로 정의했다. 표본추출은 하지 않았다.

| 지표 | 결과 |
|---|---:|
| RIFF/WAVE signature | 1,116 / 1,116 |
| decode 성공 | 1,116 / 1,116 |
| nonfinite samples | 0 |
| 총 길이 | 11,285.488 s = 3.134858 h |
| 길이 min / p50 / p95 / max | 10.000 / 10.180 / 10.242 / 10.242 s |
| codec | pcm_f32le 620, pcm_s16le 496 |
| sample rate | 16 kHz 372, 32 kHz 372, 44.1 kHz 124, 48 kHz 248 |
| channels | mono 868, stereo 248 |
| sample-weighted silence | 2.065321% |
| silence ≥ 50% files | 0 |
| sample-weighted clipping | 0.007008% |
| clipping ≥ 1% files | 2 |
| embedded-byte SHA-256 duplicate groups | 0 |
| decoded-PCM SHA-256 duplicate groups | 0 |

가장 높은 silence fraction은 MusicGen Small `id=00333`의 36.6531%다. clipping 1% 이상은 MusicGen Small `id=00489` 1.3264%, MusicGen Large `id=01022` 1.2552% 두 개다. 두 파일은 제거 대상으로 확정하지 말고, waveform 시각/청취 QA 및 peak 정규화 전후 민감도 검사 대상으로 표시한다. exact byte/PCM 중복은 없지만 perceptual fingerprint 기반 near-duplicate 검사는 아직 하지 않았다. 여기서 기술 PASS는 **파일 무결성·전수 decode PASS**를 뜻하며, perceptual equivalence까지 부정하는 표현이 아니다.

## provider shortcut

| provider | n | codec | rate / channel | 길이 | 평균 silence | clip ≥ 1% |
|---|---:|---|---|---:|---:|---:|
| AudioLDM 2 Large | 124 | f32 WAV | 16 kHz / mono | 10.000 | 0.1366% | 0 |
| AudioLDM 2 Music | 124 | f32 WAV | 16 kHz / mono | 10.000 | 0.0765% | 0 |
| MusicGen Large | 124 | f32 WAV | 32 kHz / mono | 10.180 | 0.2334% | 1 |
| MusicGen Medium | 124 | f32 WAV | 32 kHz / mono | 10.180 | 0.2938% | 0 |
| MusicGen Small | 124 | f32 WAV | 32 kHz / mono | 10.180 | 0.8723% | 1 |
| Mustango | 124 | s16 WAV | 16 kHz / mono | 10.242 | 0.1570% | 0 |
| Riffusion | 124 | s16 WAV | 44.1 kHz / mono | 10.230 | 0.0574% | 0 |
| Stable Audio v1 | 124 | s16 WAV | 48 kHz / stereo | 10.000 | 4.0298% | 0 |
| Stable Audio v2 | 124 | s16 WAV | 48 kHz / stereo | 10.000 | 3.6507% | 0 |

codec, sample rate, channel, duration 조합이 provider별로 결정적이거나 거의 결정적이다. 모델이 생성 특징 대신 수집 pipeline을 맞힐 수 있으므로 원본 표현으로 provider 분류 성능을 평가하면 안 된다. 학습 입력은 동일 sample rate/channel/길이 정책으로 canonicalize하고, 원본 codec·path·ID·shard 정보는 feature에서 제거한다. canonicalization 전후 성능 차이도 보고한다.

## 중복·누수·검증 전략

- description은 332개뿐이고, 중복 description group 270개가 1,054/1,116행을 포함한다. 동일 prompt/tag 조합이 여러 provider에 반복되므로 row random split은 prompt 누수다.
- audio path는 532개뿐이며 중복 path group 201개가 785행을 포함한다. parquet의 `audio.path`는 실제 외부 파일을 가리키는 고유키가 아니라 embedded bytes에 붙은 논리 파일명이다. 따라서 같은 path 문자열에 서로 다른 provider의 서로 다른 audio bytes가 들어갈 수 있고, path 중복과 audio SHA 중복 0은 모순이 아니다. 1,116/1,116 path가 정규화한 description으로 시작하므로 path 자체가 prompt label을 노출한다.
- ID는 provider별 연속 구간에 있다. 예를 들어 MusicGen Small은 25–489, MusicGen Medium은 507–971, Stable Audio v2는 4002–4497 범위다. ID와 shard index는 provider proxy다.
- clip을 segment로 나눌 경우 동일 원곡 segment를 split 간 공유하면 안 된다.

필수 split은 먼저 `normalized_description`을 group key로 두고, exact audio/PCM hash와 향후 perceptual duplicate component를 합친 연결성분 단위로 분리한다. 그 위에 다음 두 축을 별도로 둔다.

1. **semantic-held-out:** prompt/description group 전체를 holdout한다.
2. **provider-held-out:** generator 전체를 holdout한다.
3. **strict OOD:** 학습에 없는 provider이면서 학습에 없는 semantic group인 교집합을 최종 일반화 평가로 둔다.

provider-held-out만으로는 같은 prompt가 train/test에 들어갈 수 있고, semantic-held-out만으로는 같은 codec pipeline을 공유할 수 있다. 두 축을 섞어 하나의 random holdout 점수로 축약하지 않는다.

누수 taxonomy 재검사에서는 두 패턴이 실제로 발화했다. **Wrong null hypothesis**: `model` 열만 제거해도 ID, path, codec/rate/channel/duration이 provider를 남긴다. 해결책은 이들 필드 제거와 audio canonicalization을 함께 적용하는 것이다. **Shared-pool bias**: 여러 provider가 같은 500개 prompt pool을 공유한다. 해결책은 normalized description 연결성분을 split 이전에 고정하는 것이다.

## 라이선스·계보 판정

AIME의 명시적 CC BY 4.0 문구와 row-level provider는 Echoes의 dataset-level BY-SA 문구보다 실무적으로 강한 근거다. 이번 재판정은 (1) AIME 제작자의 직접 grant, (2) 사용자의 비영리 CC-BY/NC/SA 허용, (3) 명시적 output 재라이선스·detector-training 금지의 실제 존재를 분리했다. exact checkpoint나 tier가 없다는 사실만으로 자동 HOLD하지 않았다.

- [MusicGen 공식 model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md)는 code MIT, model weights CC BY-NC 4.0이라고 명시하지만 생성 output 재라이선스 금지는 명시하지 않는다. 사용자가 NC 자원을 허용하므로 세 MusicGen category는 GO_WITH_ATTRIBUTION이다.
- [AudioLDM2 공식 checkpoint page](https://huggingface.co/cvssp/audioldm2)는 CC BY-NC-SA 4.0이고 large와 music checkpoint를 구분하지만 생성 output 재라이선스 금지는 명시하지 않는다. 사용자가 NC·SA 자원을 허용하므로 두 category는 GO_WITH_ATTRIBUTION이다.
- [Riffusion 공식 model page](https://huggingface.co/riffusion/riffusion-model-v1)의 CreativeML OpenRAIL-M은 licensor가 output 권리를 주장하지 않는다고 명시하고 lawful use restriction을 유지한다. Riffusion은 GO_WITH_ATTRIBUTION_AND_OPENRAIL_RESTRICTIONS다.
- [Mustango 공식 checkpoint page](https://huggingface.co/declare-lab/mustango)는 Apache 2.0이다. 별도 output 금지를 찾지 못했고 AIME 직접 grant가 있으므로 GO_WITH_ATTRIBUTION이다.
- [Stability의 현재 서비스 약관](https://stability.ai/terms-of-service)은 output 권리를 사용자에게 양도하고 경쟁 서비스 학습을 제한한다. DeepVoice는 생성 모델과 경쟁하는 generator가 아니라 detector이며, 비경쟁 detector 학습을 금지하는 조항은 찾지 못했다. Stable Audio v1/v2는 GO_WITH_ATTRIBUTION이되 경쟁 generator 학습에는 사용하지 않는다. 생성 당시 tier·약관 공백은 provenance-quality 경고다.
- [Suno 현재 약관](https://suno.com/terms)은 유료와 무료/Basic output 권리를 구분하고 Services와 Output을 다른 AI/ML 모델 학습에 사용하는 것을 명시적으로 제한한다. 생성 당시 이를 뒤집는 증거가 없는 Suno v3/v3.5는 EXCLUDED_HOLD다.
- Udio는 current/contemporaneous terms를 독립적으로 확인하지 못했고 commercial provider 보수 정책에 따라 EXCLUDED_HOLD다.
- description은 CC BY-NC-SA 4.0이다. 사용자가 NC·SA 자원을 허용했으므로 선택 subset의 학습 label로 쓸 수 있지만 해당 조건과 attribution은 유지한다.

`aime-curated-manifest.csv`는 1,116행 모두를 원본 repository path, parquet SHA-256, parquet row, ID, provider, description, audio byte/PCM hash, attribution 문구와 연결한다. 이 1,116행은 `training_eligible=true`, `noncommercial_training_status=GO_WITH_AIME_CC_BY_4_ATTRIBUTION`이다. 이는 공식 문서와 사용자 정책을 결합한 프로젝트 데이터 gate이며 법률 자문은 아니다.

## 전체 audio 취득 시 전수 기준

아래 기준은 현재 curated subset의 GO를 취소하는 조건이 아니라 AIME 6,500행 **전체로 범위를 넓힐 때** 적용한다.

1. 210/210 shard의 size와 공식 LFS SHA-256 일치, parquet schema·row 수·provider purity 전수 확인.
2. 6,500/6,500 audio signature, full decode, duration/rate/channel/codec/bit depth, nonfinite, silence, clipping 전수 확인.
3. embedded-byte hash와 canonical decoded-PCM hash 전수 계산, acoustic/perceptual fingerprint로 near-duplicate component 추가.
4. prompt/description, path, ID, shard, codec shortcut을 계산하고 group split manifest를 사전에 고정.
5. MTG-Jamendo 500개는 per-track license allowlist와 attribution을 row별로 검증하고, 허용되지 않은 track을 제외.
6. Suno는 detector training을 허용하는 generation-time 약관 증거, Udio는 contemporaneous output/training rights가 확보될 때까지 다운로드·학습 대상에서 제외.
7. open-model category의 exact generation checkpoint/version은 차단 gate가 아니라 계보 완성도 필드로 보강.

## 재현 방법과 한계

`audit_aime_observed.py`는 36개 local shard를 전수 검사하고 inventory, curated manifest, shard-integrity CSV, JSON run record를 만든다. `audit_aime_repository.py`는 전체 repository projection, `plan_aime_subset.py`는 고정 subset 계획, `download_aime_subset.py`는 official LFS OID 대조 취득을 재현한다. 오디오 content 검사는 표본이 아니라 1,116행 전수다.

이번 감사가 확정하지 못한 것은 (a) 미다운로드 174 shards의 오디오 상태, (b) perceptual near-duplicate, (c) generation 당시 exact checkpoint/service tier, (d) 법률적 최종 허용 여부다. 이 한계는 현재 curated 1,116행의 attribution 조건부 GO와 전체 6,500행 미감사를 구분해서 해석해야 한다.

## 산출물

- `aime-subset-audit-run.json`: machine-readable 단일 실행 사실
- `aime-subset-audio-inventory.csv`: 1,116행 전수 오디오 inventory
- `aime-curated-manifest.csv`: provenance·attribution·rights gate 포함 manifest
- `aime-subset-shard-integrity.csv`: 36개 shard hash/size/row/provider 검증
- `aime-provider-summary.csv`: provider shortcut과 품질 요약
- `aime-rights-matrix.csv`: category별 보수적 권리 gate
- `aime-evidence-map.csv`: 주장-근거-한계 연결
- `aime-subset-acquisition-plan.csv`, `aime-subset-download-manifest.csv`, `aime-subset-download-run.json`: 취득 계획과 실제 결과
- `aime-shard-inventory.csv`, `aime-repository-audit-run.json`: 전체 repository projection
- `audit_aime_observed.py`, `audit_aime_repository.py`, `plan_aime_subset.py`, `download_aime_subset.py`: 재현 코드
