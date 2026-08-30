EXPERIMENT_BATCH: READY_FOR_FULL_TRAINING

# E01-R4 cache·numerical benchmark 보고

## 판정

strict 수치·cache·worker·runtime gate가 통과했다. 사용자에게 이 benchmark를 보고하기 전에는 장시간 3-seed 학습을 시작하지 않는다.

## 불변 계약

- manifest SHA-256: `2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6`
- manifest integrity rows: 137,328
- content-group crossing: 0
- test rows: crossing용 두 field만 사용; statistics/predictions/metrics 0
- workload: 32,768 samples/epoch × 20 epochs × 3 seeds
- segment: fixed 8초, max 8/file, explicit valid mask
- model input: waveform only; cache locator/source metadata는 feature가 아님

## strict numerical gate

- unit/adversarial tests: 15/15 PASS
- nested NaN JSON: recursive gate + `allow_nan=False` hard reject
- masked NaN logits: finite loss여도 logits guard가 hard reject
- precision: quality-first guarded FP32; CUDA autocast/GradScaler disabled
- optimizer: FP32 gradient, parameters, step counter를 batch별 검사
- cached GPU pilot: 32 guarded batches, finite loss 0.63425410, skip 0

## exact cache

- scope: non-test FMA/AIME 5,651 entries
- estimated bytes: 9,568,880,912
- actual NPY bytes: 9,557,290,108 (8.901 GiB)
- this-run build action: REUSED_COMPLETE_NO_RAW_REDECODE
- build/verify seconds: 0.000
- source locator/hash/sample-count pinned; reload max-abs-diff=0; raw originals retained
- AIME locator resolver: 1009/1009 ID assertions PASS, declared 1-based → resolved 0-based
- cache integrity: PASS

## Windows workers와 GPU

- worker 0/2/4 exact sequence: PASS
- deterministic CPU threads: parent intra/inter 1/1, worker 1
- selected workers: 2 (444.273 sample/s loader-only warm pass)
- GPU autotune batch: 32 (765.967 guarded segment/s)
- realistic cached loader+GPU: 98.662 sample/s, peak 0.373 GiB

## 보수적 full-run projection

- measured rate: 98.662 sample/s
- safety factor: 0.80
- conservative rate: 78.929 sample/s
- projected 3-seed wall time: 7.111 hours
- performance-first gate: ≤24.0 wall-hours
- status: READY_FOR_FULL_TRAINING
- arbitrary 3 GPU-hour gate는 제거했고 statistical workload는 축소하지 않았다.

## 실행 범위

- full 3-seed training started: false
- validation OOF/metric/checkpoint: 생성하지 않음
- E02: 실행하지 않음
- R1/R2/R3 code와 reports: 보존

## 재현성

- source: `experiments/e01_r4`
- code inventory SHA-256: `002e5f7bbe0ba37762f907aa975e7bfb71f95cf71393de768da1c9c54db96a16`
- config SHA-256: `626d4e61d84552f81a1294fb085ea19bff268696ac94e63ed08a4100d265756c`
- git HEAD: `4eff360b862d755fc4b06582f93740ec4bb1bda4`
