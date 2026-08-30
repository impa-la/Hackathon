EXPERIMENT_AUDIT: PASS

# DeepVoice E01-R4 independent validation audit

## 판정

E01-R4의 cache·수치 guard·worker·projection 증거를 독립 재검산했고 `READY_FOR_FULL_TRAINING` benchmark 판정을 승인한다. 이 판정은 장시간 학습 결과나 validation 성능 승인이 아니다.

## 독립 통과 사항

- R4 report/artifact JSON 13개와 cache control/index/sidecar를 strict recursive-finite 방식으로 검사했고 nested NaN hard reject를 재현했다.
- 15/15 unit/adversarial tests를 새로 실행했으며 masked NaN, skipped optimizer step, FP32-only, worker 0/2/4 검사가 포함된다.
- fresh CUDA guarded smoke와 32-batch cached pilot에서 FP32 logits/loss/gradient/parameter/step guard, skip 0을 확인했다.
- R4 source inventory 16개 byte/SHA와 digest가 run manifest에 일치한다.
- manifest SHA, 137,328 rows, split crossing 0, test crossing-only 격리를 확인했다.
- active cache 5,651 index/NPY/sidecar, 9,557,290,108 NPY bytes, float32 1-D sample counts를 전수 검사했다.
- cache/raw exact array sample 6개, AIME 1,009 ID 전수 및 raw audio file_sha256 sample 12개를 독립 검증했다.
- invalid quarantine는 active root 밖에 있으며 active index reference는 0이다. 표본 raw source hash는 manifest와 일치한다.
- worker 0/2/4 canonical tensor digest와 locator sequence가 동일하고 CPU thread pin 1/1/1, workers=2 선택을 확인했다.
- batch=32 선택과 cached end-to-end 98.6615 sample/s, safety factor 0.80을 확인했다.
- 32,768×20×3=1,966,080 training samples와 validation 18,165 segments/seed를 포함한 7.111063987시간을 재계산했다.
- full training/checkpoint/validation OOF/metric/E02 산출물은 없다.

## 승인 범위

다음 단계는 고정된 R4 code inventory와 cache index hash를 다시 gate한 뒤 full 3-seed training을 별도 immutable run으로 시작하는 것이다. 이 감사 자체는 학습 시작, checkpoint 생성 또는 성능 주장을 수행하지 않았다.
