DATA_READINESS: BLOCKED

# Echoes 리비전 데이터 감사

## 판정

현재 리비전은 기술적으로 읽을 수 있는 **생성 음악 전용 확장 데이터**이지만, 곧바로 학습에 투입할 수 있는 라벨 학습 세트는 아니다. 실제 ZIP에는 4,488개 생성 오디오와 manifest만 있고 bona-fide 오디오는 없다. README와 [Echoes 논문](https://arxiv.org/pdf/2603.23667)은 4,468개 생성 트랙과 300개 bona-fide reference의 의미적 짝을 설명하지만, 이 리비전은 reference 이름만 제공한다.

또한 README front matter와 논문은 CC BY-SA 4.0을 표시하지만 ZIP 내부에 별도 LICENSE가 없고, provider별 생성 계정 tier·출력 권리·재라이선스 권한은 증빙되지 않았다. 따라서 `echoes-curated-manifest.csv`의 4,462개 행도 모두 `training_eligible=False`로 보수적으로 고정했다.

## 범위와 방법

- 원본: `C:\Users\MY PC\Desktop\Hackathon\deepvoice\data\raw\echoes\14b0c76c6a691c42fadfab9fb6a4eb1ee8c628a2`
- 전수 범위: ZIP 4,509개 member, 추출 파일 4,489개, 오디오 4,488개. 표본 추출은 하지 않았다.
- 무결성: archive 전체 SHA-256, 중앙 디렉터리, 경로 순회, 모든 member CRC를 검사했다.
- 오디오: 모든 파일 SHA-256, 확장자/서명, FFmpeg 9.0.1 디코드, interleaved signed 16-bit PCM SHA-256, duration/sample rate/channel/codec, 무음·클리핑을 계산했다.
- 무음: `abs(sample / 32768) < 1e-4`; 클리핑: `abs(sample / 32768) >= 0.999`.
- 계보: manifest의 `original_audio`를 FMA `raw_tracks.csv` title-artist와 매핑하고 strict allowlist에 조인했다.
- 원본 보호: archive·README·manifest의 크기와 mtime을 검사 전후 비교했고 변하지 않았다. 산출물은 staging에만 기록했다.

## 원본 무결성

| 항목 | 결과 |
|---|---:|
| Echoes.zip bytes | 8,598,345,242 |
| SHA-256 | `8746dcb367f2f547399201d442ffab9121c36415815947ed4784e29b60e25b59` |
| 기대 SHA/LFS OID 일치 | 예 |
| ZIP member / file | 4,509 / 4,489 |
| ZIP uncompressed bytes | 8,843,084,266 |
| CRC 오류 / unsafe path | 0 / 0 |
| 추출 file / bytes | 4,489 / 8,843,084,266 |
| ZIP 내부 별도 LICENSE | 없음 |
| README front matter | `license: cc-by-sa-4.0` |
| 검사 전후 source state | 동일 |

추출 루트의 실질 구조는 `ATA/`, `TTA/`, `dataset_manifest.csv`뿐이다. bona-fide 오디오 디렉터리나 reference WAV/MP3는 없다.

## manifest와 실제 파일 조정

| 항목 | 수치 |
|---|---:|
| manifest 행 | 4,468 |
| manifest 고유 path | 4,464 |
| 실제 오디오 | 4,488 |
| 고유 1:1 manifest 매칭 | 4,462 |
| 중복 path 때문에 모호한 실제 파일 | 2 |
| manifest에 없는 추가 파일 | 24 |
| manifest가 가리키지만 없는 파일 | 0 |
| TTA / ATA manifest 행 | 3,165 / 1,303 |
| 실제 TTA / ATA 파일 | 3,164 / 1,324 |

두 중복 path는 다음과 같고 각각 서로 다른 러시아어 제목 3행에 연결된다.

- `ATA/musicgen/_musicgen_ATA_001.wav`
- `TTA/musicgen/_musicgen_TTA_001.wav`

24개 추가 파일 중 22개는 동일한 621바이트 ID3/MP3 헤더만 있으며 유효 MPEG frame이 없어 디코드되지 않는다. 나머지 2개는 정상 디코드되지만 manifest 계보가 없다. curated manifest는 24개 추가 파일과 두 모호 파일을 제외해 4,462행이며, FMA reference ambiguity와 모든 권리 gate를 명시적으로 보존한다.

## 오디오 전수 통계

| 항목 | 수치 |
|---|---:|
| MP3 / WAV | 3,901 / 587 |
| decode OK / 오류 | 4,466 / 22 |
| 정상 manifest 파일 decode OK | 4,462 / 4,462 |
| 총 decoded duration | 130.8171116시간 |
| duration min / p50 / p95 / max | 8.32 / 107.9728 / 233.0800 / 479.9600초 |
| manifest-duration 절대 오차 최대 | 0.06초 |
| sample rate | 16 kHz 1,149; 32 kHz 587; 44.1 kHz 1,537; 48 kHz 1,193; undecodable 22 |
| mono / stereo / undecodable | 1,736 / 2,730 / 22 |
| weighted silence fraction | 3.5598734% |
| silence >= 50% 파일 | 5 |
| weighted clipping fraction | 0.0239423% |
| clipping >= 1% 파일 | 357 |
| clipping >= 5% 파일 | 10 |

무음 50% 이상 5개는 50.02~54.64% 범위다. 클리핑 5% 이상 10개는 모두 MusicGen WAV이며 최대 6.1520%다. 이 값은 품질 판정이 아니라 정해진 sample threshold에 따른 신호 진단치다.

### 중복

- 파일 SHA-256 중복은 3그룹/26파일이다. 이 중 22파일 그룹은 위의 동일한 손상 header-only extra다.
- 유효 오디오의 exact 파일 중복은 Producer의 ATA/TTA 쌍 두 그룹, 4파일이다.
- decoded PCM SHA-256도 같은 두 그룹/4파일을 확인한다.
- perceptual fingerprint를 사용하지 않았으므로 재인코딩된 근접 중복은 이 결과에 포함되지 않는다.

## provider·codec 분리 가능성

| provider | manifest 행 | 실제 파일 | curated | extra/모호 | 형식·sample rate·channel | 권리 gate |
|---|---:|---:|---:|---:|---|---|
| acestep | 294 | 295 | 294 | 1 / 0 | MP3, 48 kHz, stereo | CONDITIONAL_HOLD |
| audioldm | 587 | 587 | 587 | 0 / 0 | MP3, 16 kHz, mono | CONDITIONAL_HOLD |
| brev | 298 | 298 | 298 | 0 / 0 | MP3, 48 kHz, stereo | HOLD |
| diffrhythm | 594 | 594 | 594 | 0 / 0 | MP3, 44.1 kHz, stereo | CONDITIONAL_HOLD |
| elevenlabs | 300 | 300 | 300 | 0 / 0 | MP3, 44.1 kHz, stereo | HOLD |
| mubert | 149 | 149 | 149 | 0 / 0 | MP3, 44.1 kHz, stereo | HOLD |
| musicgen | 591 | 587 | 585 | 0 / 2 | WAV, 32 kHz, mono | CONDITIONAL_HOLD |
| producer | 300 | 300 | 300 | 0 / 0 | MP3, 44.1 kHz, stereo | HOLD |
| songgen | 561 | 584 | 561 | 23 / 0 | 유효 파일 MP3, 16 kHz, mono | CONDITIONAL_HOLD |
| stableaudio | 194 | 194 | 194 | 0 / 0 | MP3, 44.1 kHz, stereo | HOLD |
| suno | 300 | 300 | 300 | 0 / 0 | MP3, 48 kHz, stereo | HOLD |
| udio | 300 | 300 | 300 | 0 / 0 | MP3, 48 kHz, stereo | HOLD |

provider가 형식·sample rate·channel과 거의 결정적으로 결합돼 있다. 특히 MusicGen은 WAV/32 kHz/mono, AudioLDM과 유효 SongGen은 MP3/16 kHz/mono다. 이 생성 오디오를 원본 FMA MP3와 합치면 label을 음악의 진위가 아니라 source/codec으로 맞히는 shortcut이 생길 수 있다. 평가에서는 동일한 decode/resample/channel 정책을 적용하고 codec/source별 성능을 별도로 보고해야 한다.

## FMA reference 계보

manifest의 `original_audio`는 296개 고유 문자열이다.

| 조인 결과 | generated 행 |
|---|---:|
| exact unique FMA ID | 4,255 |
| ambiguous FMA candidates | 207 |
| unmatched | 0 |
| exact하게 매핑된 고유 FMA ID | 280 |
| strict allowlist reference | 3,419 |
| strict allowlist 밖 reference | 836 |
| ambiguity로 allowlist 미결정 | 207 |

207행은 16개 reference 문자열이 복수 FMA track ID에 대응한 경우다. title 문자열만으로 임의 선택하지 않고 `reference_match_status`와 후보 ID를 curated manifest에 보존했다. FMA strict allowlist는 reference 원본 사용 가능성만 다루며, 생성 결과물의 provider 권리를 대신 증명하지 않는다.

## 라이선스 gate

### A. 추가 모델 계보 검증 후보: CONDITIONAL_HOLD

요청된 A bucket인 ACE-Step, AudioLDM, DiffRhythm, MusicGen, SongGen은 2,647개 실제 파일이며 curated 고유 파일은 2,621개다. 논문 Table II가 model family/version을 주장하지만 manifest에는 file-level model ID가 없고 exact checkpoint/API revision, weight license, 추론 서비스, 생성 당시 조건, output lineage가 재현 가능하게 고정되지 않았다. 따라서 모두 `CONDITIONAL_HOLD`다. 이 bucket 이름은 비상업 판정이 아니다. 논문 자체는 DiffRhythm과 SongGen을 commercial로 표시하므로 해당 두 provider는 권리 확인 시 commercial 조건도 함께 검증해야 한다. 예를 들어 [MusicGen 공식 model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md)는 code와 별개로 model weight를 CC BY-NC 4.0으로 설명한다. 이 사실만으로 Echoes의 MusicGen 결과물 585개를 학습 허용으로 승격할 수 없다.

### B. Commercial provider: HOLD

Brev, ElevenLabs, Mubert, Producer, Stable Audio, Suno, Udio는 1,841개이며 전부 `HOLD`다. 생성 당시 account tier와 output rights가 제공되지 않았다. 현행 [Suno Terms](https://suno.com/terms)는 유료 Pro/Premier와 무료 Basic output 권리를 다르게 다루고, [ElevenLabs Terms](https://elevenlabs.io/terms-of-use)도 무료·유료 사용 조건을 구분한다. 현재 약관은 생성 당시 약관 증명이 아니며 Echoes 저자의 tier도 알려주지 않는다.

결과적으로 curated 4,462행은 `CONDITIONAL_HOLD=2,621`, `HOLD=1,841`, `training_eligible=True=0`이다.

## 누수 방지 split 제약

1. 파일 단위 random split은 금지한다. 같은 `original_audio`를 공유하는 생성물과 추후 추가될 bona-fide reference를 하나의 `semantic_pair_group`으로 묶어야 한다. 현재 296개 semantic group이 있다.
2. exact FMA ID가 있는 경우 ID를 주 키로 쓰고, 16개 모호 reference는 수동 해소 전 같은 normalized title group으로 유지한다.
3. FMA artist·album 연결성분을 추가해 원본 계보가 train/validation 양쪽으로 넘어가지 않게 한다. 기존 FMA 감사에서 album 단위 공식 split 누수가 확인됐으므로 official split을 그대로 신뢰하지 않는다.
4. provider-held-out 평가는 provider만 떼는 축과 semantic group까지 동시에 떼는 엄격 축을 구분해 보고해야 한다. 엄격 축에서는 test provider와 test semantic group의 교차 셀만 평가하고, 해당 semantic group의 다른 provider 생성물도 train에서 제외한다.
5. 같은 PCM인 Producer ATA/TTA 두 쌍은 반드시 같은 split에 둔다. 모호한 MusicGen path 두 개와 24개 extra는 split 대상에서 제외한다.
6. codec/sample rate/channel별 subgroup 성능을 함께 보고해 provider/source shortcut을 드러낸다.

## 남은 차단 사유와 최소 조치

1. 12개 provider 각각에 대해 정확한 모델·revision/API, 생성 날짜, account tier, 당시 약관, output 소유·상업·재학습 권리를 증빙한다.
2. CC BY-SA 4.0 적용 범위가 README/metadata뿐 아니라 개별 audio binary까지 포함하며 upstream provider 조건과 충돌하지 않는다는 배포자의 증빙과 정식 LICENSE를 확보한다.
3. 300개 bona-fide reference의 실제 파일 또는 신뢰 가능한 FMA ID manifest를 확보하고, 16개 모호 reference를 수동 해소한 뒤 strict allowlist만 재감사한다.
4. `echoes-curated-manifest.csv`의 모호/extra/권리 플래그를 유지하고, 권리 증빙 전에는 어떤 행도 학습 대상으로 바꾸지 않는다.
5. 전체 DeepVoice 데이터 gate는 WaveFake 다운로드·추출·paired audit가 완료될 때까지 계속 BLOCKED다.

## 산출물

- `echoes-audit-run.json`: 정규 수치와 방법의 canonical record
- `echoes-audio-inventory.csv`: 4,488개 전수 파일 hash·join·audio 통계
- `echoes-curated-manifest.csv`: 고유 manifest 매칭 4,462행과 보수적 권리 gate
- `echoes-duplicate-groups.csv`: 파일·PCM exact duplicate 그룹
- `echoes-provider-license-summary.csv`: provider별 기술 특성과 권리 상태
- `echoes-evidence-map.csv`: 주장-근거-한계 매핑
- `audit_echoes.py`: 읽기 전용 재현 스크립트

최종 판단은 `DATA_READINESS: BLOCKED`다. 오디오 품질 문제보다 더 큰 차단점은 bona-fide 부재와 provider별 권리 계보 부재다.
