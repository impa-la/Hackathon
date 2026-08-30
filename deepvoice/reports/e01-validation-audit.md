EXPERIMENT_AUDIT: BLOCKED

# DeepVoice E01 R1 independent validation audit

감사 시각: 2026-08-30 20:57:05 +0900
범위: original `e01-*` BLOCKED_RESOURCE run only; later e01-r2/e01-r3 lineages excluded

## 판정

E01 R1은 자원 gate에서 중단한 결정 자체는 보수적으로 맞지만, 실행 건전성과 재현성 계약을 통과하지 못했으므로 `BLOCKED`다.

- finite_pilot_and_strict_json: balanced pilot is PASS despite nonfinite final_loss
- finite_pilot_and_strict_json: balanced pilot is not strict RFC JSON: non-standard JSON constant NaN
- code_report_consistency: run_e01.py differs from R1 inventory

## 핵심 차단 사유

1. `e01-balanced-pilot.json`은 `status: PASS`와 `final_loss: NaN`을 동시에 기록했다. nonfinite training loss를 PASS로 둘 수 없다.
2. R1 JSON writer가 Python 기본 `allow_nan=True` 동작을 허용해 RFC JSON이 아닌 bare `NaN` token을 저장했다. strict parser는 이 파일을 거부한다.
3. R1 run manifest가 기록한 `run_e01.py`는 13,706 bytes / SHA-256 `8ebc70...96517`이지만 감사 시 파일은 다른 크기·hash다. immutable R1 driver가 없어 exact code/report consistency를 재현할 수 없다.
4. 31.776시간 산술은 정확하지만 입력 throughput은 nonfinite-loss pilot을 PASS로 채택한 값이다. 따라서 resource 결론은 보수적 중단 근거로만 남고 유효한 E01 training preflight로 승격할 수 없다.

## 통과한 독립 검사

- manifest SHA-256, 137,328행, content-group split crossing 0
- 실제 test는 group/split crossing projection 외 사용 없음
- LJSpeech file, WaveFake ZIP, FMA MP3, AIME Parquet 네 locator를 독립 ffmpeg/pyarrow 경로로 decode하고 E01 loader sample count와 대조(PCM resampler endpoint의 최대 1 sample 반올림 차이만 허용)
- valid-sample/frame mask 및 padded-tail sentinel의 feature/logit delta 0
- 32,768-sample sampler에서 네 strata 각 8,192, speech content pairing 100%, held-out source 0, AIME provider 9종 포함
- workload 32,768×20×3 = 1,966,080 training decodes, validation 13,540 files/18,165 segments, 산출 31.776394시간
- 31.776시간은 3 GPU-hour와 24 wall-hour gate를 모두 초과하므로 full training 미실행 분기와 일치
- checkpoint, validation prediction/OOF, E01 metric/result 파일 없음; `artifacts/e01/run.json`만 존재
- tiny smoke와 pilot은 성능 결과가 아니라고 표시되어 있으며 validation 성능 주장은 없음

## 원인과 수정 책임

R1이 NaN을 PASS로 기록한 직접 원인은 최종 pilot payload에 대한 finite-loss gate와 strict JSON gate가 없었기 때문이다. benchmark 내부의 사전 loss 검사가 있었다는 주장만으로 저장된 NaN과 PASS의 모순은 해소되지 않는다. 또한 R1 이후 같은 `experiments/e01` 경로가 변경되어 원 실행 driver를 재현할 수 없다.

후속 실행은 새 immutable revision 디렉터리에서 시작하고, 매 batch 및 optimizer step 후 logits/loss/grad/parameter finite 검사를 수행하며, payload 직렬화 전에 모든 float를 검사하고 `json.dumps(..., allow_nan=False)`를 강제해야 한다. R1 산출물은 자원 중단의 참고 기록으로만 보존하고 모델 개선 근거로 사용하지 않는다.
