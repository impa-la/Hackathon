EXPERIMENT_BATCH: COMPLETE

# DeepVoice E00 evaluation-contract batch

실행 시각: 2026-08-30 19:51:04 +0900
범위: E00 scorer, label mask, split/group, content-group bootstrap, singleton-equivalence, shortcut 계약만 검증했다. 모델 학습과 E01은 실행하지 않았다.

## 판정

- E00 계약: COMPLETE
- 다음 단계: 독립 validation auditor의 `EXPERIMENT_AUDIT: PASS` 전까지 모델 개선 근거로 사용하지 않는다.
- fixture 예측: 라벨과 독립인 고정 난수이며 모델 결과, OOF 결과 또는 성능 기준선이 아니다.
- test split: group crossing 감사에만 사용했으며 예측·지표·분포 통계를 계산하지 않았다.

## 고정 입력

- manifest: `deepvoice/reports/deepvoice-training-manifest.csv.gz`
- SHA-256: `2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6`
- 행: 137,328
- validation 행: 13,540
- content group: 35,779
- seed: `20260830, 20260831, 20260832`

## 공식 scorer 계약

DACON 평가 페이지의 수식을 그대로 사용했다: `Score = 0.9 × ADS + 0.1 × CPS`; `ADS = 0.5 × (1-File EER) + 0.2 × (1-Voice EER) + 0.3 × (1-Music EER)`; `CPS = 0.5 × Voice Presence AUC + 0.5 × Music Presence AUC`.
따라서 five-head 유효 가중치는 `0.45, 0.18, 0.27, 0.05, 0.05`다. EER은 공식 `drop_intermediate=False` threshold 규칙을 재현한다. 각 head에는 AUC, EER, Brier, log loss, 15-bin ECE를 함께 기록했다.

`RobustSelectionScore`는 validation의 공식 가중 proxy와 동일하다. 모델링 계획에 수치 penalty가 정의되지 않았으므로 임의 penalty를 만들지 않았다. generator/provider/source/codec/rate/channel/duration macro·worst slice는 별도 필수 gate로 남긴다.

## label과 mask

| dataset | file fake | voice fake | music fake | voice present | music present |
|---|---:|---:|---:|---:|---:|
| LJSpeech | 0 | 0 | masked | 1 | 0 |
| WaveFake speech | 1 | 1 | masked | 1 | 0 |
| FMA real music | 0 | masked | 0 | masked | 1 |
| AIME instrumental | 1 | masked | 1 | 0 | 1 |

전체 mask audit는 20개 dataset-head 조합을 확인했다. source dataset별 정확한 관측·mask 수는 `e00-label-mask-audit.csv`에 있다.

## 통과 조건

- content group split crossing: 0개
- 실행 테스트: 7개 모두 PASS
- singleton max absolute delta: 0 (허용값 ≤ 1e-6)
- content-group bootstrap: seed별 200회 요청
- shortcut label-pure slice: 33개, 모델 성능이 아니라 데이터 교란 경보로 기록

## 재현성과 비용

- git HEAD: `7186551068e1c1c8e11b37639815fc5d5f2e8285`
- Python: `3.13.5`
- NumPy: `2.4.6`
- CUDA 사용: `False`
- 총 wall time: 60.013초
- GPU time: 0시간
- 최대 논리 CPU 기준 비용 상한: 0.200042 CPU-hour

## 산출물

- `e00-head-metrics.csv`: seed별 non-model fixture head metric
- `e00-bootstrap-summary.csv`: content-group bootstrap 95% CI
- `e00-shortcut-label-audit.csv`: train+validation shortcut label purity
- `e00-shortcut-metric-fixture.csv`: validation slice별 scorer 동작 상태
- `e00-singleton-equivalence.json`: 파일 독립 aggregation 검사
- `e00-run-manifest.json`: config/code/data/environment/runtime 버전
- `deepvoice/artifacts/e00/`: fixture 예측과 bootstrap replicate 원자료

## 제한

E00에는 학습된 예측이 없으므로 generator/provider macro와 worst 성능을 해석하지 않는다. single-domain manifest에는 실제 BOTH_PRESENT/NEITHER_PRESENT가 없으므로 joint presence와 mixed-file CPS도 검증되지 않았다. 이 결과는 scorer와 실험 계약이 실행 가능하다는 뜻이며 모델 품질을 뜻하지 않는다.

공식 근거: https://dacon.io/competitions/official/236749/overview/evaluation (확인일 2026-08-30)
