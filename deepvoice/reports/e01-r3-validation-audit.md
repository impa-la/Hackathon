EXPERIMENT_AUDIT: BLOCKED

# DeepVoice E01-R3 independent validation audit

## 판정

R3의 현재 산출물은 finite/strict이고 immutable inventory도 일치하지만, `nonfinite hard guard`와 strict serializer 계약이 구현되지 않아 `BLOCKED`다.

## 차단 사유

1. `JsonBytes`는 `allow_nan=False`를 강제하지 않는다. 동적 probe에서 `NaN`을 bare token으로 직렬화했다. 이번 출력이 우연히 finite라는 사실만으로 후속 full-run 산출물의 strict JSON을 보장할 수 없다.
2. balanced pilot은 `Loss`만 검사한다. 독립 반례에서는 masked 위치의 NaN logit과 NaN gradient가 finite loss `0.693147...` 뒤에 남았고, GradScaler는 예외 없이 step을 skip했다. `RunSeed`에는 logits/loss/gradient/parameter finite guard가 하나도 없다.
3. 따라서 run manifest의 `FP32 log-mel, GradScaler and a nonfinite hard guard` 중 FP32/GradScaler만 사실이며 hard-guard 주장은 코드보다 강하다.

## 독립 통과 사항

- R3 JSON 9개 strict parse, pilot final loss finite
- R3 source 12개 byte/SHA 및 inventory digest 완전 일치
- manifest SHA, 137,328행, crossing 0, test crossing-only 격리
- LJSpeech file, WaveFake ZIP, FMA MP3, AIME Parquet 실제 decode
- valid-length/frame mask와 padded-tail feature/logit delta 0
- CUDA autocast 내부 feature path FP32, CNN logits FP16, GradScaler enabled
- 32,768 sample에서 네 strata 각 8,192, speech pair mismatch 0, held-out 0, AIME provider 9종
- 32,768×20×3 = 1,966,080 training decodes 및 validation을 포함한 28.9100939037829시간 산술
- full training/checkpoint/validation prediction/metric 없음; R1/R2 limitation과 supersession 명시

## 수정 조건

새 immutable revision에서 serialization 전 모든 float를 검사하고 `json.dumps(..., allow_nan=False)`를 사용해야 한다. pilot과 full `RunSeed` 모두 logits, loss, unscaled gradients, optimizer step 뒤 parameters를 검사하고 nonfinite면 즉시 예외와 BLOCKED 상태를 기록해야 한다. GradScaler의 silent skipped-step은 PASS로 계산하면 안 된다.
