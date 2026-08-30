EXPERIMENT_AUDIT: PASS

# DeepVoice E00-R2 independent validation audit

감사 시각: 2026-08-30 20:13:32 +0900
범위: R2 test isolation, validation scorer/bootstrap, singleton, shortcut, provenance, R1 byte lineage

## 판정

E00-R2는 독립 감사에 통과했다. R1에서 확인된 test 통계 사용 2건이 제거되었고 E01 진입 gate를 충족한다.

## 독립 검사 결과

| 검사 | 상태 |
|---|---|
| manifest_identity | PASS |
| content_group_split_crossing | PASS |
| test_isolation_contract | PASS |
| label_mask_contract | PASS |
| official_metrics_and_brier | PASS |
| fixture_provenance | PASS |
| content_group_bootstrap | PASS |
| singleton_equivalence | PASS |
| shortcut_alert_completeness | PASS |
| r1_r2_validation_byte_equivalence | PASS |
| declared_tests | PASS |
| run_provenance_and_r1_preservation | PASS |
| independent_qa_34 | PASS |

별도 QA 체크리스트: **34/34 PASS**

## test isolation

- 실제 manifest에서 test 행은 즉시 `content_group_key`, `recommended_content_split` 두 필드로만 투영된다.
- crossing auditor는 세 번째 필드를 가진 행을 거부하고 결과에는 `crossing_group_count`만 남긴다.
- label/mask 및 shortcut label은 train+validation, scorer/bootstrap/fixture는 validation만 사용한다.
- 두 synthetic test sentinel의 모든 비허용 metadata를 서로 다른 invalid 값으로 바꿔도 retained non-test rows, crossing, validation metric과 Score가 동일했다.
- 감사 자체는 실제 test의 label, metadata, prediction 또는 성능 통계를 계산하지 않았다.

## 검증된 수치와 재현성

- manifest SHA-256과 137,328행 일치, content-group split crossing 0
- 공식 유효 가중치 0.45/0.18/0.27/0.05/0.05
- 3-seed EER, AUC, Brier, ADS, CPS, Score 독립 재계산 일치
- 3-seed × 200 content-group bootstrap artifact와 sampling digest 일치
- singleton max delta 0, 허용값 1e-6 이하
- 7개 shortcut 축의 모든 non-test label slice 및 validation slice/head 완전
- R2 validation metric/report 6개와 gzip artifact 6개가 R1과 byte-for-byte 동일
- R2 자체 테스트 9/9 PASS, 독립 QA 34/34 PASS

## 범위 제한

PASS는 E00 평가 계약의 신뢰성을 뜻한다. fixture는 label-independent uniform RNG이며 모델, OOF 또는 baseline 성능이 아니다. 따라서 E01 구현은 가능하지만 E00 수치를 모델 성능으로 해석하면 안 된다.

공식 평가 근거: https://dacon.io/competitions/official/236749/overview/evaluation (감사 확인일 2026-08-30)
