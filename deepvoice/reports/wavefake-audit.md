DATA_READINESS: READY

# WaveFake 1.2.0 최종 전수 감사

감사일은 2026-08-30 KST다. 원본 `deepvoice/data/raw/wavefake/1.2.0/generated_audio.zip`은 읽기 전용으로 검사했고 영구 해제하지 않았다. 모델링 코드는 수정하지 않았다.

## 결론

- 공식 ZIP 28,918,626,084 bytes의 MD5 `76b3e62d69f866e57ad6b1debaff434b`와 SHA-256 `431e880dc54361cbb9722f332377f20d4524d231ecfce8ce9ca187dc1b6bec30`이 검증 기록과 일치한다.
- ZIP entry는 134,266개지만 고유 생성 음원은 117,983개다. Common Voice prompt 기반 생성 WAV 16,283개가 루트와 중첩 `generated/` 경로에 바이트 단위로 두 번 저장돼 있다.
- 두 경로 모두 `gen_<id>.wav`이며 16,283쌍 모두 파일 SHA-256, PCM SHA-256, 크기, ID가 일치한다. 데이터시트도 “전부 생성 데이터이며 reference data는 재배포하지 않는다”고 명시하므로 원본·생성 쌍이 아니라 중복 사본으로 판정했다.
- 짧은 루트 경로를 canonical generated row로 유지하고 중첩 사본 16,283개를 제외했다. 그 결과 기술·권리 조건을 통과한 WaveFake 생성 행은 117,983개다.
- CRC 실패 0, WAV/PCM 파싱 실패 0, unsafe path 0, symlink 0, 비유한 sample 0이다. 원본 크기와 mtime은 감사 전후 동일하다.
- WaveFake 자체는 생성 음원만 제공한다. 실음성 LJSpeech/FMA와 결합한 최종 이진 target은 `deepvoice-training-manifest.csv.gz`의 `label`이다.

## 공식 문서와 실측 수의 조정

공식 [Zenodo v1.2.0 record](https://zenodo.org/records/5642694)는 ZIP의 MD5, 104,885개라는 초기 설명, TTS 16,283 phrases, reference data 비재배포, CC BY-SA 4.0을 게시한다. 공식 [GitHub repository](https://github.com/RUB-SysSec/wavefake)는 NeurIPS 2021 데이터셋과 Zenodo 링크를 연결한다. 공식 [datasheet PDF](https://zenodo.org/record/5642694/files/datasheet.pdf)는 117,985개, 16-bit PCM, 10 sample sets, 전부 생성 데이터라고 적는다.

| 단계 | 수치 | 해석 |
|---|---:|---|
| 논문/Zenodo 초기 설명 | 104,885 | 초기 공개 설명의 생성 clip 수 |
| 데이터시트 | 117,985 | HiFi-GAN LJSpeech 13,100개 추가 후의 문서 수치 |
| 실측 고유 생성 | 117,983 | LJSpeech 91,700 + JSUT 10,000 + TTS 16,283 |
| ZIP 전체 entry | 134,266 | 고유 생성 117,983 + TTS 중첩 중복 사본 16,283 |

`104,885 + 13,100 = 117,985`다. 데이터시트 본문의 TTS `16,285`와 실제 고유 ID `0..16,282`, 즉 16,283개 사이에는 2개 차이가 있다. 실측을 SSOT로 사용한다.

## 전수 검사 방법

- 중앙 목록의 모든 134,266 entry를 순서대로 열어 EOF까지 읽었다. Python ZIP CRC 검증 외에 CRC-32를 독립 누적했다.
- 완전한 비압축 WAV bytes와 RIFF `data` chunk PCM bytes에 각각 SHA-256을 계산했다.
- RIFF chunk를 직접 순회해 codec, sample rate, channels, bit depth, frame 수와 duration을 구했다.
- 전 sample에 대해 `abs(x) < 1e-4`를 silence, `abs(x) >= 0.999`를 clipping으로 정의했다. 표본추출은 없다.
- 250 entry마다 inventory flush와 progress JSON을 기록했다. `audit_wavefake_stream.py --resume`으로 완료 지점부터 이어갈 수 있다.

## 구조·오디오 품질

| 항목 | 실측 |
|---|---:|
| ZIP entry | 134,266 |
| canonical generated / 학습 가능 | 117,983 |
| 중첩 exact duplicate 사본 / 제외 | 16,283 |
| canonical duration | 715,159.575초 / 198.655시간 |
| raw entry duration, 중복 포함 | 778,366.359초 / 216.213시간 |
| format | 전부 mono 16-bit PCM WAV |
| canonical sample rate | 22,050 Hz 107,983개; 24,000 Hz 10,000개 |
| canonical duration min / median / p95 / max | 0.673 / 6.060 / 9.671 / 16.375초 |
| canonical weighted silence fraction | 0.019748 |
| silence 50% 이상 | 0 |
| canonical weighted clipping fraction | 1.923e-8 |
| clipping 1% 이상 | 0 |
| nonfinite sample | 0 |
| file/PCM duplicate groups | 각각 16,283, 모두 TTS 경로 이중 저장 |

생성기별 고유 수는 LJSpeech 기반 7종 각각 13,100, JSUT 기반 2종 각각 5,000, Conformer+FastSpeech2+Parallel WaveGAN TTS 16,283이다. 상세 duration·silence·clipping은 `wavefake-generator-summary.csv`에 있다.

## ID join과 누수 방지

- LJSpeech real 13,100 ID와 WaveFake 7개 생성기 각각의 13,100 ID가 완전한 1:1:7 그룹을 이룬다. 누락·extra·동일 디렉터리 중복 ID는 0이다.
- JSUT basic5000 5,000 ID는 두 생성기에서 모두 완전하다.
- TTS 16,283 ID는 두 ZIP 경로에 정확히 반복된다. nested copy는 학습·검증·테스트 전부에서 제외한다.
- random row split이나 잘라낸 segment 단위 split은 금지한다. LJSpeech ID, JSUT ID, TTS prompt ID를 indivisible content group으로 사용해야 한다.
- LJSpeech 원본은 과거 MP3 계보를 갖고 생성 파일은 generator별 주파수 지문을 갖는다. codec/rate/source shortcut을 통제하되 변환은 split 이후 fold 내부에서만 수행한다.
- content split 외에 generator-held-out 및 source-family-held-out 평가를 별도로 보고해야 한다.

## 권리 조건

WaveFake 생성 데이터는 CC BY-SA 4.0이다. 재사용 시 저자·데이터셋·라이선스를 표시하고, 변경 여부를 밝히며, adapted material을 공유할 때 ShareAlike를 지킨다. Common Voice upstream reference audio는 ZIP에 존재하지 않는다. canonical TTS generated row에도 WaveFake attribution을 유지한다.

## 재현·산출물

- 실행 SSOT: `wavefake-audit-run.json`
- 전수 행: `wavefake-audio-inventory.csv.gz`
- 생성기 요약: `wavefake-generator-summary.csv`
- ID coverage: `wavefake-pairing-summary.csv`
- 중복 근거: `wavefake-duplicate-groups.csv.gz`
- LJSpeech join: `wavefake-ljspeech-pairing.csv.gz`
- 그룹: `wavefake-source-group-manifest.csv.gz`
- 재현 코드: `audit_wavefake_stream.py`

대형 CSV는 GitHub 단일 파일 한계를 피하려고 `mtime=0` 결정적 gzip으로 배포한다. `package_large_csv.py`가 같은 gzip을 재생성하고 `large-csv-package-run.json`이 압축 전후 SHA-256을 기록한다.
