# DeepVoice 외부 데이터 취득 계획

- 기준일: 2026-08-30
- 상태: `DATA_READINESS: READY` (Phase 1 취득·무결성·권리·전수 감사 완료)
- 원칙: 원본 archive와 license/terms를 `deepvoice/data/raw/<source>/<version>/`에 보존하고 SHA-256을 기록한다.
- 저장공간: WaveFake 원본과 압축 감사 산출물 보존 후 C 드라이브 여유 약 47 GiB. WaveFake는 디스크 중복을 피하려고 ZIP 내부를 전수 스트리밍 감사했다.

## Phase 1

| Source | Version | Official URL | Exact bytes | SHA-256 / integrity | Status | Purpose |
|---|---:|---|---:|---|---|---|
| LJSpeech | 1.1 | https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 | 2,748,572,632 | `be1a3045...e3de8aa5` | 다운로드·안전 해제·전수 감사 완료 | WaveFake 영어 subset과 출처가 맞는 real speech control |
| FMA metadata | official current archive | https://os.unil.cloud.switch.ch/fma/fma_metadata.zip | 358,412,441 | `d9527a52...736bf265` | 다운로드·CRC·내부 SHA1·전수 감사 완료 | track별 license·artist·split 감사 |
| FMA small | official current archive | https://os.unil.cloud.switch.ch/fma/fma_small.zip | 7,679,594,875 | `f923bbef...a5d02c` | 다운로드·안전 해제·8,000개 전수 감사 완료 | real music 30초 샘플의 codec·장르·권리 분포 감사 |
| Echoes | HF revision `14b0c76c...628a2` | https://huggingface.co/datasets/Octavian97/Echoes | 8,598,345,242 | `8746dcb3...e25b59` | 다운로드·안전 해제·4,488개 전수 감사 완료, **학습 HOLD** | fake music 후보의 품질·권리 계보 감사 |
| AIME | HF revision `b84d4be5...f2601` | https://huggingface.co/datasets/disco-eth/AIME | published 62,281,475,287; selected 1,246,815,838 | selected 36 LFS SHA-256 전부 일치 | 공개 모델 9종 × 4 pure shards, 1,116행 취득·전수 오디오 감사 완료, attribution 조건 학습 가능 | CC BY 4.0 fake music 후보 |
| WaveFake | 1.2.0 | https://zenodo.org/records/5642694/files/generated_audio.zip?download=1 | 28,918,626,084 | MD5 `76b3e62d69f866e57ad6b1debaff434b`; SHA-256 `431e880d...b6bec30` | 다운로드·공식 MD5·134,266 entry 전수 감사 완료; 고유 생성 117,983개 | 다중 vocoder fake speech |

각 source의 공식 license/datasheet도 같은 디렉터리에 저장한다. FMA는 현재 source/license가 확인되는 CC0/BY/BY-SA/BY-NC/BY-NC-SA 트랙만 allowlist하고 ND/custom/restricted/unknown을 제외한다. WaveFake 파생 mix는 BY-SA 호환 계층으로 격리한다. Echoes의 dataset card와 논문이 CC BY-SA 4.0을 표시하더라도, 실제 ZIP의 11개 생성 제공자별 exact checkpoint/API revision·계정 tier·출력물 권리 증빙이 없으므로 4,462개 curated 샘플을 모두 `training_eligible=False`로 유지한다. AIME는 저자가 생성한 6,000개 트랙에 CC BY 4.0을 직접 부여했지만, 이번 취득에서는 권리 불확실성을 더 줄이기 위해 Suno·Udio를 제외하고 공개 모델 9종의 pure-provider shards만 골랐다.

## Phase 2

ASVspoof 2015 음원은 공식 DataShare split archive 합계 약 22.4 GB다. 현재는 WaveFake 압축본·해제본의 동시 보존과 C 드라이브 안전 여유를 우선하므로 추가 취득하지 않는다. WaveFake 감사 후에도 필요하면 외장 드라이브 또는 추가 저장 위치를 먼저 정한다.

## 감사 완료 전 금지

- 모델 구조 확정 및 학습
- source/track/speaker/generator가 섞인 임의 분할
- raw archive 삭제 또는 덮어쓰기
- 실제 license가 확인되지 않은 음원을 학습 manifest에 등록
