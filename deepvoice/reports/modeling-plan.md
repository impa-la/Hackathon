MODEL_PLAN: READY

# DeepVoice 모델링 계획

작성 기준일: 2026-08-30 (Asia/Seoul)
범위: 모델 설계, 실험 순서, 검증·추론·권리 게이트. 이 문서는 학습이나 모델 구현을 수행하지 않는다.

## 1. 진입 조건과 고정 계약

최종 데이터 게이트는 `deepvoice-final-data-readiness.md`의 `DATA_READINESS: READY`이다. 이전 감사 문서에 남은 중간 `BLOCKED` 표시는 최종 감사와 evidence map으로 해소되었으며, 모델링의 유일한 데이터 SSOT는 아래 manifest이다.

- manifest: `deepvoice-training-manifest.csv.gz`
- manifest SHA-256: `2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6`
- 총 137,328행, 268.412시간
- real 18,229행, synthetic 119,099행
- 고정 split: train 110,059 / validation 13,540 / test 13,729
- content group 35,779개, split 교차 0개

manifest의 `recommended_content_split`은 변경하지 않는다. 모든 샘플·segment·augmentation은 원본 행의 split과 content group을 상속한다. 임의 행 분할, segment 분할, test 재분할을 금지한다.

대회 출력은 정확히 다음 5개 확률이다.

| head | 의미 | 최종 Score 기여도 |
|---|---|---:|
| FILE_FAKE_PROB | 파일 단위 합성 여부 | 0.45 |
| VOICE_FAKE_PROB | 음성 합성 여부 | 0.18 |
| MUSIC_FAKE_PROB | 음악 합성 여부 | 0.27 |
| VOICE_PRESENT_PROB | 음성 존재 여부 | 0.05 |
| MUSIC_PRESENT_PROB | 음악 존재 여부 | 0.05 |

공식 Score는 `0.9 × ADS + 0.1 × CPS`이다. 제출 환경은 Ubuntu 22.04, Python 3.11.15, NVIDIA L4 22.4 GiB, CPU 6개, RAM 28 GiB, 설치 10분, 1,200파일 추론 60분, ZIP 10 GiB, 완전 오프라인, 파일 독립 추론이다.

## 2. 검증 계약을 먼저 고정한다

모델 선택 전에 다음 축을 고정하고 이후 변경하지 않는다.

1. **content split:** manifest의 train/validation/test를 그대로 사용한다.
2. **generator-held-out:** WaveFake generator 하나씩을 train에서 제외하고 동일 validation content ID의 해당 generator를 평가하는 shadow fold를 만든다. LJSpeech 1 real : 7 fake pairing을 group 단위로 보존한다.
3. **provider-held-out:** AIME provider 하나씩 제외하는 shadow fold를 만든다. AIME fake와 FMA real의 source-label confounding 때문에 이 결과는 음악 일반화의 보조 증거로만 사용한다.
4. **source-family-held-out:** JSUT, TTS, AIME를 각각 source-family OOD stress set으로 취급한다. fake-only set은 고정 threshold의 recall/FNR만 보고 EER이나 종합 선택 점수로 과장하지 않는다.
5. **FMA component:** artist-album 연결 성분을 content group으로 유지한다.
6. **AIME content:** normalized description hash group을 유지하여 provider/prompt 중복이 split을 건너지 못하게 한다.
7. **shortcut slice:** codec, sample rate, channel, duration, source family별 metric과 예측 분포를 항상 함께 기록한다.

test split은 최종 후보와 calibration을 동결한 뒤 한 번만 연다. seed는 `20260830, 20260831, 20260832` 세 개이고, 신뢰구간 bootstrap 단위는 행이 아니라 content group이다.

모델 선택용 `RobustSelectionScore`는 validation의 공식 가중 head score를 기본으로 하되, 각 head에서 content / held-out generator·provider / source-family stress 축의 macro 값과 worst-slice 값을 함께 기록한다. 후보 승격 조건은 기본 score 개선뿐 아니라 held-out macro EER 악화가 0.01을 넘지 않는 것이다. source-label shortcut로만 얻은 점수는 승격 근거가 아니다.

## 3. 선택 구조

### 3.1 핵심안

`frozen WavLM-Base-Plus + frozen MERT-v1-95M + lightweight artifact CNN + task-specific gated fusion + five masked heads`

| branch | 입력과 역할 | 선택 이유 |
|---|---|---|
| Speech semantic/acoustic | Microsoft WavLM Base Plus, frozen; layer-weighted hidden states의 mean/std pooling | 16 kHz 음성 표현에 특화되고 94.7M급이라 L4 예산에 맞는다. LJS/WaveFake speech 축을 담당한다. |
| Music acoustic | M-A-P MERT-v1-95M, frozen; layer-weighted mean/std pooling | 음악 표현에 특화되어 FMA/AIME 축의 WavLM 편향을 보완한다. 95M급으로 dual-base 구성이 가능하다. |
| Artifact | LFCC/log-linear filterbank, delta/delta2, energy-weighted phase 및 instantaneous-frequency residual을 작은 LCNN/ResNet으로 처리 | generator vocoder, codec, phase 불연속 등 저수준 위변조 흔적을 semantic encoder와 독립적으로 포착한다. |
| Fusion | 각 branch를 256차원 adapter로 투영한 뒤 task별 soft gate와 gated concatenation | speech/music/head별로 branch 신뢰도를 다르게 학습하되 hard routing 실패를 피한다. |
| Heads | FILE_FAKE, VOICE_FAKE, MUSIC_FAKE, VOICE_PRESENT, MUSIC_PRESENT의 독립 logit head | 공식 제출 계약과 직접 정렬한다. 관측되지 않은 label은 loss mask로 제외한다. |

WavLM과 MERT는 첫 실험군에서 완전히 frozen한다. 이는 작은 데이터의 source shortcut 과적합을 낮추고, 원 checkpoint의 라이선스 경계를 분명히 하며, 학습·추론 자원을 제한한다. encoder unfreeze나 LoRA는 frozen fusion이 검증을 통과한 뒤 권리 검토를 다시 하는 조건부 실험이다.

### 3.2 unified 후보를 핵심에서 제외한 이유

- **BEATs:** speech/music 통합 acoustic encoder로 유망하지만 official checkpoint의 획득 파일, SHA-256, checkpoint 자체의 명시적 권리 문구가 아직 고정되지 않았다. 권리와 파일 재현성이 확인되기 전에는 핵심안에 넣지 않는다.
- **PANNs Cnn14:** repo code는 MIT이고 제공 데이터에 exact checkpoint도 있으나 Zenodo weight의 별도 라이선스 grant가 명시되지 않았다. 권리 확인 전에는 routing/fallback에도 사용하지 않는다.
- **HTS-AT:** code는 MIT지만 Google Drive weight의 정확한 hash와 별도 재배포 권리가 불명확하다.
- **XLS-R 300M:** Apache-2.0이라 권리 fallback으로 좋지만 300M급 단일 speech encoder로, dual-base 핵심안보다 runtime과 패키지 비용이 크고 music specialist가 아니다.

## 4. 입력과 segment

1. MP3/WAV/FLAC를 파일별로 decode한다. 파일명, 경로, codec tag, provider ID 같은 metadata는 feature로 사용하지 않는다.
2. 모든 신호를 mono float waveform으로 만들고 deterministic peak safety만 적용한다. 원 loudness 자체를 지우는 전역 정규화는 하지 않는다.
3. canonical waveform은 16 kHz이다. WavLM과 artifact branch는 이를 직접 사용한다.
4. MERT 입력은 canonical 16 kHz waveform을 24 kHz로 다시 resample한다. 원본 44.1 kHz FMA만 직접 24 kHz로 보내지 않는다. 이 규칙은 train/test bandwidth 및 sample-rate shortcut을 차단한다.
5. segment는 8초이고 파일당 최대 8개다. 8초 이하는 pad+valid mask, 긴 파일은 시작/끝을 포함한 균등한 deterministic 위치를 사용한다. 60초 파일도 최대 8개다.
6. 전체 파일을 먼저 합쳐 batch inference하지 않는다. padding batch가 필요하면 파일 경계를 보존하고 singleton-equivalence test를 통과해야 한다.

artifact의 primary view는 공식 test의 16 kHz 조건과 정렬한다. 다만 고주파 vocoder/codec 흔적을 너무 일찍 버리는지 확인하기 위해 **모든 파일에 동일하게 적용하는 fixed 32 kHz LFCC/phase view**를 조건부 ablation으로 둔다. 이 view는 원 decode waveform을 32 kHz로 만들고, 원본이 16 kHz이면 동일 resampler로 upsample한다. 다음 세 조건을 모두 만족할 때만 채택한다: (a) 16 kHz low-pass 대조군보다 held-out generator/provider가 개선됨, (b) source/sample-rate/codec probe의 예측력이 증가하지 않음, (c) 공식 16 kHz stress set이 악화되지 않음. 조건을 못 맞추면 16 kHz artifact view를 유지한다.

## 5. pooling, fusion, file 논리

각 encoder는 선택된 hidden layer의 학습 가능한 convex weight로 time representation을 만들고 valid frame의 mean/std를 연결한다. encoder 출력은 LayerNorm과 256차원 adapter를 지난다. artifact CNN도 256차원으로 맞춘다.

task (t)별 gate는 세 branch embedding으로부터 `softmax(g_t)`를 만들고, weighted branch embedding과 원 branch의 저차원 projection을 연결한다. gate entropy와 branch-dropout을 약하게 사용해 한 source shortcut으로 붕괴하는 것을 감시한다.

segment logit은 파일별로 고정된 `mean logit + top-k mean logit` 혼합으로 집계한다. 비율과 k는 validation에서 한 번 정하고 고정한다. 파일 fake의 논리 보조 확률은 다음과 같다.

`p_logic = 1 - (1 - p_voice_present × p_voice_fake) × (1 - p_music_present × p_music_fake)`

최종 FILE_FAKE는 direct file head와 `p_logic`의 validation-fixed alpha blend다. direct head와 logic head 사이에는 logit-space symmetric consistency를 두고, 존재하지 않는 domain의 conditional fake probability가 file fake에 기여하지 않도록 `p_voice_fake × p_voice_present`, `p_music_fake × p_music_present`의 곱으로만 결합한다. CPS 보조 제약은 (a) `p_voice_fake ≤ p_voice_present + margin`, (b) `p_music_fake ≤ p_music_present + margin`, (c) 관측된 single-domain row에서 FILE_FAKE와 해당 conditional fake가 일치하도록 한다. 각 항은 관련 존재/진위 label이 모두 관측된 row에만 mask하여, 알 수 없는 voice/music fake label을 가짜 음성/음악 negative로 만들지 않는다.

## 6. label mask

현재 자료가 직접 증명하는 label만 학습한다.

| source | file fake | voice fake | music fake | voice present | music present |
|---|---:|---:|---:|---:|---:|
| LJSpeech real | 0 | 0 | masked | 1 | 0 |
| WaveFake speech | 1 | 1 | masked | 1 | 0 |
| FMA real music | 0 | masked | 0 | masked | 1 |
| AIME instrumental fake music | 1 | masked | 1 | 0 | 1 |

AIME는 공식 논문이 “10 seconds and instrumental versions only”라고 설명하므로 voice present=0으로 둘 수 있다. FMA에 voice가 없다고 가정하지 않는다. JSUT/TTS는 speech fake와 voice present label을 갖되 source-family OOD 선택 구간에서는 훈련에 넣지 않는다.

현재 manifest는 single-domain row뿐이다. 실제 `BOTH_PRESENT`와 `NEITHER_PRESENT`가 없으므로 두 presence head의 joint identifiability와 mixed-file CPS는 현재 데이터만으로 입증할 수 없다. 이 공백을 숨기지 않고 validation report의 한계로 고정한다.

그러므로 mixed-presence를 조용히 생성하거나 기존 manifest에 끼워 넣지 않는다. 조건부 on-the-fly mixing은 manifest license field가 **CC0, CC BY, 또는 서로 호환되는 CC BY-SA stratum**으로 확인된 source끼리만 허용한다. NC와 SA처럼 결합 저작물의 조건이 충돌하거나 해석이 불분명한 waveform 조합은 금지한다. derived 파일의 source pair, mixing gain, lineage, 양쪽 content group을 기록하고, 새 manifest와 별도의 `DATA_READINESS: READY` 재감사를 받은 뒤에만 E10을 실행한다. `NEITHER_PRESENT`용 배경음도 같은 권리·lineage gate를 통과해야 한다.

## 7. loss와 불균형 처리

각 head의 관측 label에만 class-balanced BCE-with-logits를 사용한다. 특히 voice/music fake loss는 해당 진위 label과 domain 존재가 직접 관측된 row에서만 계산한다. head별 loss는 관측 mask 수로 정규화한 뒤 공식 Score 기여도 `0.45, 0.18, 0.27, 0.05, 0.05`를 적용한다. differentiable file/conditional consistency와 CPS hinge 보조 loss의 초기 계수는 각각 0.05와 0.02이며, calibration 또는 held-out 성능을 해치면 0으로 중단한다.

초기에는 focal loss를 쓰지 않는다. 119,099 synthetic 대 18,229 real 불균형에는 도움이 될 수 있지만 calibration을 훼손할 수 있기 때문이다. class-balanced sampling과 effective-number weight로 먼저 비교한다. label smoothing도 E00 scorer와 calibration 동작을 확인하기 전에는 쓰지 않는다.

sampling은 batch 기준 speech 50%, music 50%를 목표로 한다.

- speech: real/fake 1:1. LJS content ID를 먼저 뽑고, 한 epoch에서 7 fake generator 중 하나를 회전 선택하여 동일 content의 1:7 중복이 gradient를 지배하지 못하게 한다.
- music: FMA real/AIME fake 1:1. AIME provider를 균등 sampling하고 FMA artist-album component를 group 단위로 sampling한다.
- JSUT/TTS: 모델 선택 중에는 held-out stress로 유지한다. 핵심 구조가 동결된 뒤에만 final training 포함 여부를 별도 실험한다.

## 8. augmentation

augmentation은 split 뒤에 waveform에서 수행하고 원본 group을 상속한다. label과 source 양쪽에 대칭으로 적용한다.

- gain, mild EQ, low-level noise, 짧은 RIR
- resample round-trip, MP3/AAC/Opus 계열의 허용 codec round-trip
- telephone band-limit와 μ-law 계열
- 약한 clipping과 dynamic range 변형

artifact signal을 모두 지우지 않도록 no-op/weak 비율을 충분히 둔다. source마다 서로 다른 codec을 주거나 FMA만 stereo로 유지하는 식의 label-correlated augmentation을 금지한다. augmentation 강도는 shortcut slice 악화를 기준으로 중단한다.

## 9. 학습 단계

1. **계약 baseline:** 기존 log-mel CNN과 scorer를 재현해 five-head mask, group metric, singleton inference를 검증한다.
2. **artifact-only:** LFCC+phase branch로 semantic encoder 없이 shortcut과 OOD 신호를 측정한다.
3. **specialist single branch:** frozen WavLM, frozen MERT를 각각 측정한다.
4. **dual specialist fusion:** 두 encoder와 artifact branch를 gated fusion한다. 이것이 핵심 후보이다.
5. **robustness:** symmetric codec/telephone augmentation, CQT, fixed 32 kHz LFCC/phase artifact ablation을 비교한다.
6. **calibration/runtime:** aggregation, temperature, file logic alpha를 validation에서 고정하고 1,200-file worst-case benchmark를 통과시킨다.
7. **조건부:** BEATs/PANNs/HTS-AT 권리 확인, mixed-data 재감사, LoRA/partial unfreeze의 권리·runtime 재검토 후에만 실행한다.

실험별 가설, 성공/중단 기준은 `experiment-plan.csv`가 SSOT다.

## 10. calibration과 metric

각 head는 validation logit에 scalar temperature 또는 bias-free Platt scaling을 비교한다. provider/source별 calibrator는 사용하지 않는다. FILE_FAKE alpha, segment mean/top-k blend, threshold는 모두 validation에서 고정한다.

EER/AUC 같은 rank metric은 단조 temperature만으로 개선되지 않는다는 점을 기록한다. calibration의 목적은 CPS와 cross-head logic 정합성이다. test 파일 집합의 평균, 분위수, class prior로 보정하는 batch adaptation은 파일 독립 규정 때문에 금지한다.

보고 metric:

- head별 ROC-AUC, EER, log loss/Brier, calibration error
- 공식 가중 proxy 및 RobustSelectionScore
- generator/provider macro와 worst group
- codec/rate/channel/duration/source slice
- content-group bootstrap 95% CI와 3-seed 평균/표준편차

## 11. 추론과 자원

핵심 경로는 FP16 inference, encoder frozen, batch 최대 8 segment, 파일당 segment cap 8이다. 1,200×8=9,600 segment, dual encoder 최대 19,200 encoder segment pass이다.

목표 예산은 총 2,680초(44.7분)로 15분 이상의 여유를 둔다. 상세 항목과 성능 gate는 `inference-budget.md`에 고정한다. 목표 peak는 GPU 10 GiB 이하, RAM 12 GiB 이하, 두 원 weight 합계 약 720 MiB(755 MB), 제출 ZIP 1.5 GiB 이하이다.

decode는 native torchaudio/soundfile을 우선하고 파일마다 ffmpeg process를 만들지 않는다. CPU decode worker는 4개로 제한한다. dependency wheel과 custom code를 패키지에 포함하고 import smoke test를 Ubuntu/Python 3.11에서 수행한다. 설치 중 network 접근은 0건이어야 한다.

## 12. 라이선스와 제출 패키징

권리의 정확한 SSOT는 `pretrained-model-rights-matrix.csv`다.

- WavLM Base Plus model card의 공식 license link는 UniSpeech의 CC BY-SA 3.0이다. unilm code의 MIT와 checkpoint 권리를 혼동하지 않는다.
- MERT-v1-95M은 CC BY-NC 4.0이다. 비영리 대회 사용은 사용자 허용 범위지만 attribution, license notice, 변경 고지가 필요하다.
- WavLM, MERT, 학습한 adapter/head는 **서로 분리된 checkpoint/module 파일**로 유지한다. 모든 tensor를 불투명한 한 파일에 병합하지 않는다.
- `THIRD_PARTY_NOTICES`, 각 원문 license, official URL, revision, 파일 SHA-256, 변경 내역을 포함한다.
- MERT는 `trust_remote_code`를 runtime에 사용하지 않는다. exact revision의 `configuration_MERT.py`, `modeling_MERT.py`를 vendor하고 source hash와 upstream notice를 고정한다.
- encoder를 fine-tune/LoRA하면 derived checkpoint에 원 license가 어떻게 적용되는지 제출 전 다시 판정한다. 판정 전에는 frozen encoder+별도 head만 허용한다.
- 추론 ZIP에는 실행에 필요한 checkpoint/code/notice만 넣고 raw 268.412시간 audio나 optimizer state는 넣지 않는다.
- 대회 2차 검증에서 요구하는 actual training manifest, split, config, seed, source lineage와 학습에 실제 사용한 파일 목록은 **별도 검증 전달물**로 보존·제출한다. 이는 10 GiB 추론 ZIP에 원 학습 데이터를 모두 넣는다는 뜻이 아니다.

## 13. fallback과 중단 기준

권리 fallback:

1. MERT 또는 WavLM의 조합/재배포 조건을 충족하지 못하면 해당 branch를 제거한다.
2. Apache-2.0이 명확한 XLS-R 300M + artifact branch를 권리 fallback으로 측정한다.
3. 모든 third-party weight를 제외해야 하면 artifact-only를 최종 안전 fallback으로 둔다.

runtime fallback:

1. segment cap 8→6을 비교하고 RobustSelectionScore 감소가 0.005 이하일 때만 적용한다.
2. branch-dropout으로 학습한 뒤, validation에서 확실한 presence에만 적용되는 deterministic branch router를 시험한다. singleton-equivalence와 worst-slice를 모두 통과해야 한다.
3. single WavLM+artifact는 core 대비 RobustSelectionScore 감소가 0.01 이하일 때만 runtime fallback으로 허용한다.

공통 중단 조건:

- content group 교차 또는 manifest 변경 발견
- 권리 matrix가 `READY_CORE`가 아닌 weight를 핵심 경로에 사용
- held-out macro EER 0.01 초과 악화
- 3-seed 표준편차 0.005 초과
- L4 worst-case inference 50분 초과 또는 peak GPU 18 GiB 초과
- test batch 통계나 파일명/provider metadata 의존

## 14. 구현 착수 승인 조건

다음을 모두 만족하면 구현을 시작할 수 있다.

- [x] 최종 데이터 `DATA_READINESS: READY`
- [x] manifest와 split hash 고정
- [x] five-head label mask 정의
- [x] 핵심 checkpoint revision/hash와 권리 상태 기록
- [x] 60분 inference budget과 segment cap 정의
- [ ] 법적 고지가 포함된 실제 제출 package dry run
- [ ] L4에서 E00 scorer 및 1,200-file synthetic worst-case benchmark

마지막 두 항목은 구현 단계 gate이며 본 설계의 `MODEL_PLAN: READY`를 막지 않는다.

## 공식 근거

- DACON 대회: https://dacon.io/competitions/official/236749/overview/description
- WavLM checkpoint: https://huggingface.co/microsoft/wavlm-base-plus
- WavLM-linked license: https://github.com/microsoft/UniSpeech/blob/main/LICENSE
- MERT checkpoint: https://huggingface.co/m-a-p/MERT-v1-95M
- XLS-R checkpoint: https://huggingface.co/facebook/wav2vec2-xls-r-300m
- BEATs official repo: https://github.com/microsoft/unilm/tree/master/beats
- PANNs official repo: https://github.com/qiuqiangkong/audioset_tagging_cnn
- PANNs Zenodo checkpoint record: https://zenodo.org/records/3987831
- HTS-AT official repo: https://github.com/RetroCirce/HTS-Audio-Transformer
- AIME paper: https://arxiv.org/html/2506.19085
