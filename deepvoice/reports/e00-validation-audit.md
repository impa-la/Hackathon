EXPERIMENT_AUDIT: BLOCKED

# DeepVoice E00 independent validation audit

감사 시각: 2026-08-30 20:01:50 +0900
범위: E00 scorer, label/mask, split policy, content-group bootstrap, singleton, shortcut, provenance records

## 판정

E00 결과는 모델 개선 근거로 채택할 수 없다. 고정 stop criterion인 `test statistic use`가 실제 코드와 산출물에서 확인됐다.

- test_statistics_nonuse: run_e00.py line 514 computes dataset/head row, observed, masked and observed-value summaries from AllRows, which includes the locked test split
- test_statistics_nonuse: contract.py records test_row_count in the split summary although the fixed policy permits only crossing detection

## 독립 검사 결과

| 검사 | 상태 | 핵심 근거 |
|---|---|---|
| manifest_identity | PASS | SHA-256 고정, 137,328행 |
| content_group_split_crossing | PASS | 전체 content_group의 split crossing 0 |
| label_mask_contract | PASS | train+validation에서 4개 source mapping 일치 |
| official_metric_and_brier | PASS | DACON EER/AUC/ADS/CPS/Score와 Brier 독립 재계산 일치 |
| content_group_bootstrap | PASS | 3 seed × 200회 artifact 및 sampling digest 일치 |
| singleton_equivalence | PASS | 3 seed 모두 max delta ≤1e-6 |
| fixture_provenance | PASS | validation-only, label-independent RNG, non-model/non-OOF 표시 |
| shortcut_alert_completeness | PASS | 7개 축, 모든 slice/head 및 HIGH/MONITOR 경보 완전 |
| run_provenance | PASS | seed/config/git/code SHA/environment/runtime 기록 일치 |
| test_statistics_nonuse | BLOCKED | 위반 2건: test row count 저장, AllRows label/mask summary |

## test split 계약 위반

E00의 고정 정책은 test를 content-group crossing 탐지에만 사용하는 것이다. 그러나:

1. `contract.py`의 `AuditGroupCrossings`가 `test_row_count`를 계산해 run manifest에 저장했다.
2. `run_e00.py`가 `SummarizeLabelMasks(AllRows, AllLabels, AllMasks)`를 호출해 test를 포함한 dataset/head별 행 수, 관측 수, 마스크 수, 관측 label 값을 `e00-label-mask-audit.csv`에 저장했다.

이는 예측 지표를 test에서 계산하지 않았다는 주장만으로 해소되지 않는다. `experiment-plan.csv`의 E00 stop criterion은 더 넓은 `test statistic use`이며, `modeling-plan.md`는 최종 후보와 calibration 동결 전 test를 열지 않도록 고정한다. 따라서 BLOCKED다.

감사 자체는 test label·예측 분포를 새로 계산하지 않았다. 전체 row count와 content-group split membership으로 crossing만 확인했고, label/mask·metric·shortcut 검사는 train+validation 또는 validation에 한정했다.

## 통과한 계약

- manifest SHA와 137,328행이 고정 입력과 일치한다.
- content-group split crossing은 0이다.
- 4개 데이터셋의 five-head label/mask mapping이 train+validation에서 일치한다.
- 공식 가중치는 0.45/0.18/0.27/0.05/0.05이고, 저장된 3-seed EER, AUC, ADS, CPS, Score, Brier를 독립 재계산해 일치했다.
- content_group_key bootstrap 3-seed × 200회가 저장 artifact와 일치하고 같은 seed의 sampling digest가 재현됐다.
- singleton delta는 3-seed 모두 0이며 허용치 1e-6 이하다.
- fixture는 validation-only이며 seed와 row count만 받는 uniform RNG와 정확히 일치한다. 모델, OOF, baseline이 아니라는 표시는 config, run record, row artifact에 존재한다.
- shortcut 감사 7개 축과 모든 slice/head가 존재하고 label-pure slice는 HIGH로 누락 없이 표시됐다.
- seed, config, git HEAD/status, 실행 코드 SHA, 환경, runtime/cost 기록이 존재하며 현재 파일과 일치한다.

## 수정 책임과 재감사 조건

실험 엔지니어가 기존 결과를 수정하지 말고 새 E00 run으로 다음을 수행해야 한다.

- full-manifest label/mask 생성을 제거하고 train+validation 또는 validation으로만 제한한다.
- split crossing 함수는 test count를 반환·저장하지 않고 group→split 집합의 crossing 여부만 기록한다.
- 이전 `e00-label-mask-audit.csv`, run manifest, batch report를 성능 근거로 폐기하고 새 run ID/별도 출력 경로로 재실행한다.
- 재감사 전까지 E01 및 후속 모델 개선을 시작하지 않는다.

공식 평가 근거: https://dacon.io/competitions/official/236749/overview/evaluation (감사 확인일 2026-08-30)
