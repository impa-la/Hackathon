# DeepVoice split·누수 제약

## SSOT 규칙

`deepvoice-training-manifest.csv.gz`의 `content_group_key`를 기준으로 train/validation/test를 분리한다. 동일 key를 둘 이상의 fold에 넣거나, 먼저 segment로 자른 뒤 segment를 무작위 분배하면 안 된다.

결정적 할당은 salt `deepvoice-final-content-group-split-v1`과 group key를 `SHA-256`으로 해시한 뒤 앞 64 bit를 80% train, 10% validation, 10% test에 배정한다. segment가 필요하면 원본 행의 split을 상속한다.

| source_family | indivisible group | 이유 |
|---|---|---|
| ljspeech | `ljspeech:<LJ-ID>` | real과 WaveFake 7개 vocoder 출력의 동일 내용·ID 결합 |
| jsut_basic5000 | `jsut_basic5000:<ID>` | 두 generator의 동일 문장 결합 |
| common_voice_prompt | `common_voice_prompt:<prompt-ID>` | 동일 prompt 기반 TTS; nested exact copy는 제외 |
| fma_music | `fma:<artist-album connected-component>` | artist와 album 재등장 누수 방지 |
| aime_music | `aime-prompt:<semantic hash>` | 같은 description/prompt의 provider 간 재등장 방지 |

## 평가 축

- primary paired speech: LJSpeech real 13,100개와 같은 ID의 WaveFake LJS synthetic 91,700개. ID 단위 content split과 class-balanced sampling/metric을 사용한다.
- content-group split: 기본 성능과 튜닝용. 전체 manifest 행을 무가중 혼합한 단일 점수는 금지한다.
- generator/provider-held-out: 보지 못한 생성기 일반화.
- source-family-held-out: speech/music, LJSpeech/JSUT/TTS/FMA/AIME 계보 일반화. 단, 한 label만 존재하는 source-family는 독립적인 counterpart 확보 전까지 stress test로만 해석한다.
- source-stratified report: dataset, generator, codec, sample rate, channels, duration bin별 지표.

## 금지·주의

- 행·segment 무작위 split 금지.
- 전체 데이터로 normalization 통계를 먼저 계산하는 행위 금지.
- WaveFake nested `generated/` TTS 16,283개 로딩 금지.
- filename, absolute path, parquet shard/index, provider-coded AIME ID를 feature로 사용 금지.
- source마다 다른 sample rate/channel/codec을 그대로 label 대리 변수로 쓰지 않도록 검증한다.
- real 18,229 대 synthetic 119,099의 class imbalance를 그대로 accuracy에 반영하지 않는다. balanced accuracy, per-class recall 등 class-balanced 지표와 fold 내부 sampling을 사용한다.
- FMA real-only 대 AIME synthetic-only를 한 점수로 비교하면 dataset shortcut이 label과 일치한다. 같은-domain real counterpart 없이 이를 일반 deepfake 성능으로 해석하지 않는다.
- AIME semantic hash는 normalized prompt/description exact identity다. 의미 임베딩 기반 near-duplicate 군집은 아니므로 유사 prompt stress test를 별도로 둔다.
- resampling, loudness normalization, padding/cropping 정책은 fold가 정해진 뒤 적용하고 원본 계보별 결과도 함께 보존한다.
