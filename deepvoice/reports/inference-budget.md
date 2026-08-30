# DeepVoice 60분 추론 예산

상태: `INFERENCE_BUDGET: READY_FOR_BENCHMARK`
대상: NVIDIA L4 22.4 GiB, CPU 6개, RAM 28 GiB, Ubuntu 22.04, Python 3.11.15
한도: 1,200파일 / 60분, offline, 파일 독립, 제출 ZIP 10 GiB

## 1. 고정 workload

- 입력 파일 수: 1,200
- 파일 길이: 4–60초
- segment 길이: 8초
- 파일당 deterministic cap: 8 segment
- 최대 segment: `1,200 × 8 = 9,600`
- core encoder: WavLM Base Plus + MERT-v1-95M
- 최대 encoder segment pass: `9,600 × 2 = 19,200`
- artifact pass: 최대 9,600
- inference dtype: FP16, accumulation/pooling FP32
- encoder는 frozen/eval, gradient와 activation checkpointing 없음
- batch: 파일 내 segment 최대 8. 파일 간 batching은 singleton-equivalence가 정확할 때만 허용

이 계산은 모든 파일이 60초인 보수적 상한이다. 평균 길이가 짧다는 가정은 합격 조건에 쓰지 않는다.

## 2. 3,600초 배분

| 단계 | 목표 초 | 누적 초 | 근거와 gate |
|---|---:|---:|---|
| process 시작, import, weight hash/load | 90 | 90 | 모든 weight를 로컬에서 읽고 network 0건 |
| input discovery, sample_submission 검증 | 10 | 100 | 1,200개 ID/경로와 5개 열 확인 |
| decode, mono, resample, segment | 480 | 580 | CPU worker 4개, 평균 0.40초/file 이하 |
| WavLM inference | 780 | 1,360 | 9,600 segment에서 평균 12.3 segment/s 이상 |
| MERT inference | 900 | 2,260 | 9,600 segment에서 평균 10.7 segment/s 이상 |
| LFCC/phase artifact branch | 180 | 2,440 | 평균 53.3 segment/s 이상 |
| segment/file aggregation, five heads, CSV | 60 | 2,500 | 파일별 정적 aggregation, schema 검증 |
| allocator/I/O/queue 여유 | 180 | 2,680 | 측정되지 않은 overhead 포함 |
| **예비** | **920** | **3,600** | 총 15분 20초 |

목표 총시간은 2,680초(44.7분)다. 승격 hard gate는 3,000초(50분) 이하여서 최소 10분의 예비를 남긴다. 공식 절대 한도는 3,600초이나, 3,000초를 넘는 모델은 최종 후보로 승격하지 않는다.

## 3. throughput gate

microbenchmark가 아닌 end-to-end worst-case를 최종 기준으로 한다.

| 항목 | 최소 gate | 목표 |
|---|---:|---:|
| 전체 파일 처리 | 0.40 file/s (3,000초) | 0.45 file/s (2,680초) |
| decode/resample | 2.5 file/s | 3.0 file/s |
| WavLM 8초 segment | 12 segment/s | 15 segment/s |
| MERT 8초 segment | 10 segment/s | 13 segment/s |
| artifact | 40 segment/s | 53 segment/s |
| CSV/write | 20 file/s | 40 file/s |

각 branch는 warm-up 20개 파일 후 200개 worst-case 파일로 측정하고, 마지막에는 1,200개 60초 corpus를 한 process에서 실행한다. decode와 GPU compute가 겹치더라도 표의 예산은 겹침을 가정하지 않은 보수적 합으로 보고한다.

## 4. memory와 package

| 자원 | core 추정/목표 | 승격 hard gate | 공식 한도 |
|---|---:|---:|---:|
| raw pretrained weights | 약 720 MiB(755 MB) | 1.5 GiB | ZIP 전체 10 GiB |
| 제출 ZIP | 1.5 GiB 이하 목표 | 8 GiB 이하 내부 gate | 10 GiB |
| peak GPU | 10 GiB 이하 목표 | 18 GiB | 22.4 GiB |
| peak RAM | 12 GiB 이하 목표 | 20 GiB | 28 GiB |
| install | 120초 이하 목표 | 600초 | 600초 |

두 encoder는 동시에 GPU에 올릴 수 있지만, memory 변동을 줄이기 위해 branch별 sequential forward와 embedding buffer 재사용을 기본으로 한다. batch 8에서 OOM이면 batch 4→2로 내리되 시간 gate를 다시 측정한다. 모든 allocation은 `inference_mode` 아래에서 수행한다.

WavLM, MERT, trainable head는 별도 파일이다. 원 checkpoint를 중복 압축하거나 optimizer state, training cache, raw dataset을 **추론 ZIP에 포함하지 않는다**. 대회 2차 검증에서 요구하는 실제 학습 파일·manifest·lineage는 별도 검증 전달물로 준비하며, 10 GiB 추론 ZIP에 268.412시간 raw audio를 넣는다는 뜻이 아니다.

## 5. decode와 전처리

- native torchaudio/soundfile decode를 우선하고 파일마다 ffmpeg subprocess를 띄우지 않는다.
- format fallback이 꼭 필요하면 bundled binary 하나와 license를 고정하고 worker pool에서 호출 수를 제한한다.
- CPU 6개 중 4개를 decode/resample, 1개를 main/aggregation, 1개를 OS/I/O 여유로 둔다.
- canonical waveform은 mono 16 kHz다.
- WavLM: 16 kHz waveform
- MERT: canonical 16 kHz를 동일 resampler로 24 kHz 변환
- artifact primary: 16 kHz LFCC/phase
- artifact 32 kHz ablation을 채택하면 전처리/branch 예산 180초 안에서 다시 합격해야 한다.
- 8초 미만 pad에는 valid mask를 적용하고 pooling에서 pad frame을 제외한다.
- 8개 위치는 파일 길이만으로 deterministic하게 정하고, 다른 test 파일의 정보에 의존하지 않는다.

## 6. offline/install gate

제출 bundle에 다음을 고정한다.

- Python 3.11용 wheelhouse와 lockfile
- exact checkpoint 파일, revision, SHA-256
- MERT exact-revision custom code와 로컬 SHA-256
- `THIRD_PARTY_NOTICES`와 license 원문
- model config, head schema, preprocessing constants

검증은 빈 network namespace 또는 outbound 차단 환경에서 한다.

1. 새 virtual environment에서 로컬 wheel만 설치
2. import smoke test
3. 모든 weight와 vendored source hash 검증
4. 한 파일 inference와 schema check
5. 1,200 worst-case end-to-end 실행

다운로드 시도, `trust_remote_code`, cache miss에 따른 hub 접근이 한 번이라도 발생하면 실패다.

## 7. 파일 독립성과 singleton-equivalence

같은 파일을 다음 세 방식으로 실행했을 때 5개 확률의 최대 절대 차이가 `1e-6` 이하여야 한다.

1. 파일 하나만 단독 실행
2. 서로 다른 길이/format 파일 앞뒤에 배치
3. 1,200파일 목록의 순서를 뒤집어 실행

BatchNorm running update, test-batch normalization, dynamic class prior, provider/source별 calibration, 전체 batch percentile threshold를 금지한다. 결과 순서는 sample_submission ID로만 정렬한다.

## 8. runtime fallback 순서

fallback은 validation과 runtime을 함께 통과한 사전 제작 artifact만 사용한다. 실행 도중 남은 시간에 따라 모델을 바꾸지 않는다.

1. **segment cap 8→6:** RobustSelectionScore 감소 0.005 이하, worst held-out EER 악화 0.005 이하일 때 채택. 최대 segment는 7,200으로 25% 절감한다.
2. **deterministic presence router:** branch-dropout으로 미리 학습하고, 아주 높은 presence confidence에서만 불필요 specialist를 생략한다. singleton-equivalence와 all-domain stress를 통과해야 한다.
3. **WavLM+artifact single branch:** core score 차이 0.01 이하일 때만 미리 정한 runtime package로 사용한다.
4. **artifact-only safety:** third-party weight의 제출 권리가 마지막 단계에서 해소되지 않을 때의 권리 안전 fallback이다.

XLS-R 300M은 권리 fallback이지 runtime fallback이 아니다. WavLM보다 큰 weight와 compute 때문에 시간 초과를 해결하지 못한다.

## 9. benchmark 기록 schema

각 run은 다음을 저장한다.

`run_id, git_commit, package_sha256, device, driver, torch_version, files, total_audio_seconds, segments, decode_seconds, wavlm_seconds, mert_seconds, artifact_seconds, aggregate_seconds, total_seconds, files_per_second, peak_gpu_bytes, peak_ram_bytes, install_seconds, network_attempts, singleton_max_abs_delta, status`

최종 `PASS` 조건:

- install ≤600초
- total ≤3,000초 내부 gate
- peak GPU ≤18 GiB
- peak RAM ≤20 GiB
- ZIP <10 GiB
- network_attempts=0
- singleton_max_abs_delta≤1e-6
- 1,200행과 정확한 five-head schema
