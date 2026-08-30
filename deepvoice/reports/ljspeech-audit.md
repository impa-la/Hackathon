DATA_READINESS: BLOCKED

# LJSpeech 1.1 전수 오디오 감사

감사 시각은 2026-08-30 KST이며, 원본은 읽기 전용으로 검사했다. 이 단계에서 LJSpeech 실음성은 완전하지만 WaveFake 생성 오디오가 아직 다운로드 중이고 FMA 오디오 감사도 끝나지 않았으므로 학습 데이터 준비 상태는 `BLOCKED`다. 모델링 코드와 원본 파일은 수정하지 않았다.

## 결론

- LJSpeech 1.1 아카이브는 2,748,572,632 bytes이며 SHA-256은 `be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5`다. 제공된 공식 기대값과 일치한다.
- 추출본은 13,102 files, 3,801,111,440 bytes다. 구성은 WAV 13,100개, `metadata.csv`, `README`이며, 하위 `wavs` 디렉터리 1개와 데이터셋 루트까지 세면 검증 extractor의 13,104 members와 일치한다.
- WAV 13,100/13,100개가 모두 정상 파싱됐고 메타데이터 13,100행과 ID 기준 1:1 완전 조인된다. 누락·고아 ID·중복 ID는 모두 0이다.
- 전 WAV가 mono, 22,050 Hz, 16-bit, uncompressed PCM이다. 총 길이는 86,117.0763초, 즉 23.9214시간이다.
- 파일 전체 SHA-256 중복과 PCM SHA-256 중복은 모두 0그룹이다. 완전 무음 파일은 0개다.
- `abs(sample/full_scale) < 1e-4` 기준 전체 무음 표본 비율은 2.2275%다. 파일별 최대는 13.3066%이고 50% 이상 파일은 0개다.
- `abs(sample/full_scale) >= 0.999` 기준 클리핑 표본은 전체 1,898,881,532개 중 2개뿐이다. `LJ007-0073`, `LJ038-0029`에서 각 1표본이며, 심각한 전역 클리핑 징후는 아니다.
- 원문/정규화 문장이 정확히 같은 중복 텍스트 그룹은 10개, 영향 ID는 36개다. 이 ID들은 오디오가 다르더라도 반드시 같은 split 그룹으로 묶어 문장 누수를 막아야 한다.
- 로컬 LJSpeech README는 원 녹음이 128 kbps MP3에서 왔기 때문에 MP3 artifact가 남을 수 있다고 명시한다. 현재 컨테이너가 PCM WAV라는 사실은 이 계보 artifact가 없다는 뜻이 아니다.
- WaveFake 로컬 datasheet는 생성 WAV 117,985개, 네트워크별 디렉터리, 권장 split 없음, 참조 실음성 미재배포를 명시한다. 최종 실행 중 `generated_audio.zip`은 2,157,289,472 bytes에서 2,165,186,560 bytes로 증가해 SHA-256 계산 중에도 변했다. 따라서 기록된 부분 파일 해시는 유효한 원본 해시로 채택하지 않았고, 추출 오디오는 현재 0개다.

## 원본 무결성·비변경 확인

| 항목 | 실제 값 | 판정 |
|---|---:|---|
| `LJSpeech-1.1.tar.bz2` bytes | 2,748,572,632 | 존재 |
| archive SHA-256 | `be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5` | 기대값 일치 |
| 추출 file count | 13,102 | extractor 결과 일치 |
| 추출 bytes | 3,801,111,440 | extractor 결과 일치 |
| WAV bytes | 3,798,339,464 | 전수 합계 |
| 감사 전후 archive/metadata/README size·mtime | 동일 | 원본 비변경 |

아카이브는 tar.bz2라 ZIP CRC 개념을 적용하지 않았다. 기존 verified extractor가 보고한 `MemberCount 13,104 / FileCount 13,102 / UncompressedBytes 3,801,111,440`와 추출본 합계를 교차 확인했고, 이번 스크립트는 재추출하지 않았다.

## 방법

표본추출 없이 WAV 13,100개 전부를 읽었다. 각 파일에 대해 완전 파일 SHA-256과 PCM data chunk SHA-256, 헤더, 프레임 수, 길이, peak, RMS, 무음 표본 수, 클리핑 표본 수를 스트리밍 계산했다. 블록 크기는 262,144 frames다.

- 무음: `abs(sample / 32768) < 0.0001`
- 클리핑: `abs(sample / 32768) >= 0.999`
- 파일 중복: WAV 전체 bytes의 SHA-256 동일
- 오디오 중복: WAV data chunk PCM bytes의 SHA-256 동일
- 텍스트 그룹: 정규화 문장을 Unicode casefold한 뒤 연속 공백을 하나로 줄이고 SHA-256 계산
- random seed: 사용하지 않음. 전수 검사이며 확률적 연산이 없다.

이 임계값은 운영상 재현 가능한 진단 규칙이지 지각적 무음/왜곡의 절대 정의가 아니다. MP3 계보 artifact는 코덱 헤더만으로 판정할 수 없으며, 본 감사는 별도 청취·PESQ/POLQA 판정을 하지 않았다.

## 메타데이터 스키마·조인

`metadata.csv`는 quote escaping이 없는 pipe-delimited 텍스트로 읽었다. 큰따옴표는 인용부호가 아니라 문장 문자이므로 `csv.QUOTE_NONE`을 사용한다.

| 필드 | 의미 | 결측 | 중복/이상 |
|---|---|---:|---:|
| ID | 대응 WAV stem | 0 | 중복 ID 0, 형식 이상 0 |
| transcription | 원문 | 0 | 정확 중복 10 groups / 36 rows |
| normalized transcription | 숫자·단위 등을 확장한 정규화문 | 0 | 정확 중복 10 groups / 36 rows |

- 유효 행 13,100, malformed 0
- WAV→metadata 조인 13,100/13,100
- WAV without metadata 0, metadata without WAV 0
- transcription과 normalized transcription이 다른 행 1,505
- 비 ASCII 문자를 하나 이상 포함한 원문 행 119. 이 중 파운드 기호 `£`가 107회라 README의 “19 transcriptions” 설명과 단순 비 ASCII 행 집계는 정의가 다르다.
- binary deepfake target은 이 파일에 없다. LJSpeech 전 행은 실음성 `real` provenance이며, WaveFake와 결합한 manifest에서만 `real/fake` target을 만들어야 한다.

## 오디오 분포

| 지표 | 값 |
|---|---:|
| count / parse OK / parse error | 13,100 / 13,100 / 0 |
| total duration | 86,117.0763 sec = 23.9214 h |
| mean / std | 6.5738 / 2.1853 sec |
| min / p01 / p05 | 1.1101 / 1.6557 / 2.6774 sec |
| p25 / p50 / p75 | 4.9878 / 6.7641 / 8.3895 sec |
| p95 / p99 / max | 9.7247 / 10.0265 / 10.0962 sec |
| duration > 10 sec / < 1 sec | 190 / 0 files |
| file-level silence fraction p50 / p95 / max | 1.9443% / 4.1252% / 13.3066% |
| sample-weighted silence fraction | 2.2275% |
| all-silent / silence >= 50% | 0 / 0 files |
| peak p50 / p95 / max | 0.56856 / 0.81056 / 0.99957 |
| RMS p50 / p95 / max | 0.06525 / 0.08313 / 0.13099 |
| clipped files / samples | 2 / 2 |
| whole-file duplicate groups | 0 |
| PCM duplicate groups | 0 |

파일별 전체 수치와 해시는 `ljspeech-audio-inventory.csv`, 중복 텍스트 그룹은 `ljspeech-duplicate-groups.csv`에 있다.

## MP3 artifact와 전처리 누수 위험

LJSpeech README의 계보는 “원 LibriVox 녹음이 128 kbps MP3로 배포됐고 인코딩 artifact가 있을 수 있음”이다. 현재 모든 파일이 PCM WAV라는 헤더 검사는 계보 artifact를 제거하거나 부정하지 못한다. WaveFake datasheet는 원 음성에서 mel spectrogram을 추출해 여러 vocoder로 재생성했다고 설명하므로, real/fake 사이 artifact·대역폭·노이즈바닥 차이가 쉬운 지름길이 될 수 있다.

검증 원칙은 다음과 같다.

1. split을 먼저 고정한 뒤 train 안에서만 gain, resampling, codec, trimming augmentation 파라미터를 정한다.
2. real/fake 양 클래스에 동일한 codec·sample-rate·loudness·duration 처리 확률을 적용한다.
3. 경로, 파일명, vocoder 디렉터리, RIFF metadata, byte size 같은 계보 필드는 feature에서 제외한다.
4. 원본 그대로의 성능과 codec-normalized 성능을 함께 보고 차이가 큰지 확인한다.
5. 전체를 MP3로 한 번 더 변환하는 것만으로 누수 제거를 주장하지 않는다. 기존 MP3 계보와 생성기별 주파수 artifact가 남을 수 있다.

## WaveFake 동일 ID pairing·split 설계

현재 pairing 실측은 `generated_audio.zip`이 다운로드 중이고 추출 오디오가 0개라 실행 불가다. 다운로드가 끝난 뒤 다음 gate를 통과해야 한다.

1. ZIP 크기와 mtime이 해시 전후 동일한지 확인하고, 전체 SHA-256·CRC·경로순회·확장자-서명 검사를 통과시킨다.
2. datasheet대로 네트워크 디렉터리를 provenance로 보존하고, 파일명에서 `LJ\d{3}-\d{4}`를 추출한다. 문서가 실제 filename convention까지 보장하지 않으므로 실제 트리로 규칙을 검증한다.
3. LJSpeech ID마다 real 1개와 각 LJSpeech-reconstruction generator의 fake를 조인한다. missing, extra, duplicate `(generator, LJ_ID)`를 모두 별도 표로 낸다.
4. TTS의 novel phrases는 같은 LJSpeech 원문에 대한 짝이 아니므로 paired benchmark에서 제외하고 별도 OOD test로 둔다.
5. split group은 단순 파일이 아니라 연결 요소다. 같은 `LJ_ID`의 real과 모든 fake를 묶고, 정규화 문장이 중복인 10개 그룹/36개 ID도 추가로 합친다.
6. 이 그룹 단위로 train/validation/test를 예: 80/10/10으로 배치하되, 클래스와 generator 분포를 그룹 수준에서 맞춘다. 재현 seed와 최종 ID manifest를 고정한다.
7. LJSpeech는 화자 1명이라 speaker-disjoint 검증이 불가능하다. 별도 다화자 real speech corpus를 추가해 unseen-speaker·unseen-generator test를 별도로 둔다. FMA 음악은 cross-domain music 검증용이지 unseen-speaker 대체재가 아니다.
8. 랜덤 파일 split 성능은 채택하지 않는다. 동일 문장/원본에서 나온 real과 여러 fake가 서로 다른 split에 들어가면 내용·길이·운율이 사실상 답안 역할을 한다.

## 검증 가설과 현재 상태

| 가설 | 검정 | 현재 결과 |
|---|---|---|
| H1: 공식 아카이브가 완전하다 | SHA-256 기대값 일치 | 통과 |
| H2: WAV와 transcript가 완전 조인된다 | ID set equality | 통과, 13,100 = 13,100 |
| H3: 오디오 포맷이 균일하다 | 전수 WAV 헤더 | 통과, 전부 mono/22.05 kHz/16-bit PCM |
| H4: 손상·완전무음·중복이 없다 | 전수 decode/무음/두 종류 SHA-256 | 통과 |
| H5: 심각한 clipping이 없다 | 전수 표본 임계값 | 2 files에 각 1표본; 경미한 관찰 |
| H6: 같은 문장이 split을 넘지 않는다 | canonical normalized text group | 설계 필요, 10 groups/36 IDs |
| H7: real/fake 동일 ID pairing이 완전하다 | WaveFake 실제 파일 조인 | 미검증/차단 |
| H8: codec artifact가 label을 대신하지 않는다 | 대칭 전처리·raw vs normalized 비교 | 모델 단계 검증 필요 |

## 남은 차단 사유와 최소 조치

1. WaveFake `generated_audio.zip` 다운로드를 완전히 끝낸다. 완료 전 partial SHA-256은 폐기한다.
2. 안정된 ZIP을 안전 검증·추출한 뒤 실제 generator/ID pairing 표를 만든다.
3. FMA small MP3 8,000개를 메타데이터·라이선스 allowlist와 조인해 오디오 품질과 사용 가능 수를 확정한다.
4. 위 두 단계가 끝난 뒤에만 group-stratified split manifest를 고정하고 모델링을 시작한다.

## 재현 산출물

- `audit_ljspeech.py`: 전수 읽기 전용 감사 코드
- `ljspeech-audit-run.json`: 기계 판독 가능한 집계와 실행 근거
- `ljspeech-audio-inventory.csv`: 13,100개 전수 파일별 해시·헤더·길이·음질·조인 상태
- `ljspeech-duplicate-groups.csv`: 텍스트/파일/PCM 중복 그룹
- `ljspeech-evidence-map.csv`: 주장-근거-검정 연결표

집계 SSOT는 `ljspeech-audit-run.json`, 파일별 SSOT는 `ljspeech-audio-inventory.csv`다. 이 Markdown과 evidence map의 숫자는 사람이 읽기 위한 snapshot 요약이며, 이후 다운로드 중인 WaveFake ZIP의 크기가 변해도 자동 갱신되지 않는다.
