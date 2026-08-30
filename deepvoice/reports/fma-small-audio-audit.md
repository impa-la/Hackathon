DATA_READINESS: BLOCKED

# FMA small 공식 오디오 전수 감사

2026-08-30 KST 기준 공식 `fma_small.zip`과 추출 MP3를 읽기 전용으로 검사했다. FMA small 실음악 오디오는 확보됐지만 WaveFake 생성 오디오·real/fake pairing manifest가 아직 완성되지 않았으므로 전체 프로젝트는 `BLOCKED`다. 모델링 코드와 원본 파일은 수정하지 않았다.

## 결론

- 아카이브는 7,679,594,875 bytes, SHA-256 `f923bbef327820456d50965c3c320c3c7b6dab8456449429fd78f7ec96a5d02c`이며 기대값과 일치한다.
- ZIP 8,002 members는 MP3 8,000개, `checksums`, `README.txt`로 구성된다. 경로순회 후보 0, 총 비압축 크기 7,975,472,258 bytes다. 추출본의 8,002 files와 byte 합이 정확히 일치한다.
- MP3 ID 8,000개는 curated/raw metadata의 small ID 8,000개와 완전 1:1 조인된다. 누락·고아·중복 ID는 모두 0이다.
- 내장 SHA-1 checksum은 8,000/8,000 일치하고 전체 파일 SHA-256 중복은 0그룹이다.
- FFmpeg decode는 7,997/8,000 성공했다. `99134`, `108925`, `133297`은 ID3 tag는 있으나 연속 MPEG frame이 없고 decode가 실패한다. 이 중 `133297`은 strict allowlist에 포함되므로 허용 5,130개 중 실제 decodable 수는 5,129개다.
- decoded PCM SHA-256 중복은 12그룹/24트랙이다. 12그룹 모두 같은 공식 split과 같은 artist-album 연결성분 안에 있지만, 4그룹은 허용/제외 라이선스가 섞여 있다. 오디오가 같아도 허용 ID만 사용하고 그룹 전체를 같은 split에 둬야 한다.
- strict allowlist 5,130개는 모두 실제 파일로 존재하며 checksum도 모두 일치한다. 라이선스 분포는 BY-NC-SA 2,884, BY-NC 930, BY 893, BY-SA 369, CC0 54다.
- 30초에서 0.1초 이내인 파일은 7,994개다. decode 성공 파일 중 `98565` 1.6076초, `98567` 0.5105초, `98569` 1.5292초가 1초 이상 벗어나며 모두 strict allowlist다. decode 실패 3개는 길이를 산출할 수 없다.
- 전체 음원은 MP3지만 sample rate·채널·bitrate가 균일하지 않다. source/codec shortcut 차단 없이 다른 fake corpus와 섞으면 포맷이 라벨을 대신할 위험이 크다.

## 감사 방법

표본 없이 8,000개 전부를 검사했다. 각 파일은 SHA-1/SHA-256, 확장자와 ID3/MPEG 연속 frame signature를 확인한 뒤 FFmpeg 8.0.1로 signed 16-bit little-endian PCM에 decode했다. decoded PCM을 저장하지 않고 스트리밍 결과만 집계했다.

- 파일 중복: 전체 MP3 bytes SHA-256 동일
- 음향 중복: 같은 FFmpeg 버전이 출력한 interleaved s16le PCM SHA-256 동일
- 무음: `abs(sample / 32768) < 1e-4`
- 클리핑: `abs(sample / 32768) >= 0.999`
- 길이: decoded sample count / (sample rate × channels)
- random seed: 없음. 전수·결정론적 검사다.
- 안전 상한: 파일 하나가 120초를 넘는 PCM을 출력하지 않게 제한했으며 해당 상한 도달 0, 180초 wall timeout 0이다.

decoded PCM 해시는 decoder/version에 의존하는 로컬 exact fingerprint다. 지각적으로 같은 재인코딩을 모두 찾는 acoustic fingerprint는 아니며, 근사 중복 탐지는 후속 별도 검사가 필요하다.

## 무결성·조인

| 항목 | 실제 값 | 판정 |
|---|---:|---|
| archive bytes | 7,679,594,875 | 기대값 일치 |
| archive SHA-256 | `f923bbef...a5d02c` | 기대값 일치 |
| ZIP members / MP3 | 8,002 / 8,000 | 기대값 일치 |
| ZIP/extracted bytes | 7,975,472,258 / 7,975,472,258 | 일치 |
| unsafe ZIP path | 0 | 통과 |
| checksum entries / matches | 8,000 / 8,000 | 통과 |
| MP3 ↔ curated metadata | 8,000 / 8,000 | 완전 조인 |
| MP3 ↔ raw metadata | 8,000 / 8,000 | 완전 조인 |
| strict allowlist present | 5,130 / 5,130 | 완전 존재 |
| allowlist classifier mismatch | 0 | 통과 |
| 감사 전후 source size·mtime | 동일 | 원본 비변경 |

공식 checksum 일치는 공식 ZIP에 실린 bytes라는 뜻이지 음향 decode 가능성을 보장하지 않는다. 실제로 checksum은 일치하지만 3개 파일은 ID3-only/비정상 짧은 파일이라 decode되지 않는다.

## 포맷·길이

| 지표 | 실제 값 |
|---|---:|
| MP3 / decode OK / error | 8,000 / 7,997 / 3 |
| codec | MP3 7,997, 판정 불가 3 |
| sample rate | 44,100 Hz 7,572; 48,000 Hz 411; 22,050 Hz 14; 불가 3 |
| channels | stereo 7,912; mono 85; 불가 3 |
| stream bitrate 상위 | 320 kb/s 3,748; 256 1,485; 192 1,048; 128 363; 160 226; 기타/VBR 1,127 |
| total decoded duration | 239,729.0042 sec = 66.5914 h |
| duration min / median / max | 0.5105 / 29.9766 / 30.0147 sec |
| within ±0.1 sec of 30 | 7,994 |
| >1 sec deviation / undecodable | 3 / 3 |
| MP3 file size min / median / max | 1,298 / 996,162.5 / 1,205,694 bytes |

짧은 3개와 decode 실패 3개는 별도 quarantine 대상이다. 30초에 가까운 길이만으로 정상성을 단정하지 않고 checksum·decode·signature를 함께 사용해야 한다.

## 무음·클리핑

| 지표 | 전체 | strict allowlist |
|---|---:|---:|
| silence sample fraction, sample-weighted | 0.24236% | inventory로 필터 가능 |
| silence ≥50% files | 3 | 1 |
| clipping sample fraction, sample-weighted | 0.09743% | inventory로 필터 가능 |
| any clipped sample files | 3,598 | 별도 필터 필요 |
| clipping ≥1% files | 96 | 64 |
| clipping ≥5% files | 41 | 29 |
| clipping ≥10% files | 26 | 20 |

`72059`는 strict allowlist이지만 silence fraction 99.9996%, peak 0.000183으로 실질적 near-silent 파일이다. 제외 subset의 `110736`, `119979`도 각각 silence 91.69%, 69.23%다. high-clipping은 source mastering의 실제 특성일 수 있어 일괄 삭제 근거로 삼지 않았지만, 상위 파일은 `fma-small-audio-inventory.csv`의 clipping 열로 반드시 검토·민감도 분석해야 한다.

## strict allowlist의 실제 오디오 상태

| 항목 | 수 |
|---|---:|
| metadata allowlist | 5,130 |
| 파일 존재/checksum/signature | 5,130 / 5,130 / 5,129 |
| decode OK | 5,129 |
| decode 실패 | 1 (`133297`) |
| 30초에서 >1초 편차 | 3 (`98565`, `98567`, `98569`) |
| near-silent ≥50% | 1 (`72059`) |
| PCM duplicate groups fully allowed | 8 |
| PCM duplicate mixed allowed/excluded | 4 |
| split | training 4,117 / validation 550 / test 463 |

따라서 5,130은 권리 metadata gate를 통과한 “파일 존재 수”이고, 자동 학습 투입 가능 수는 적어도 decode 실패 `133297`을 빼 5,129 이하로 봐야 한다. 짧은 3개, near-silent 1개, 고클리핑 파일의 정책을 검증 전에 고정해야 하며 임의 삭제 후 성능만 비교하면 selection leakage가 된다.

## 중복·원본 계보

전체 MP3 bytes가 동일한 파일은 없지만, decoded PCM이 정확히 같은 쌍이 12개다. 대체로 같은 artist·title이 다른 album ID로 재수록된 경우다. 예를 들어 `114236↔125820`, `144938↔145707`, `45102↔51785`가 해당한다.

- PCM duplicate 12그룹은 모두 공식 split을 넘지 않는다.
- 모두 같은 artist-album 연결성분 안에 있다.
- 4그룹은 strict allowlist 여부가 섞인다. 허용 파일과 bytes가 다르더라도 decoded audio가 같으므로 제외 파일을 추가 데이터로 되살리면 라이선스 gate를 우회하게 된다.

## album+artist 연결성분 split 제약

small 8,000개로 `artist_id -- album_id` bipartite graph를 만들었다.

| 지표 | 값 |
|---|---:|
| connected components | 1,760 |
| 공식 split을 넘는 components | 66 |
| 영향 tracks | 1,072 |
| largest component | 208 tracks / 40 artists / 41 albums |
| allowlist 중 교차 component 소속 | 736 tracks / 51 components |

공식 split은 artist ID만 보면 겹침이 없지만 compilation album이 여러 artist를 이어 component가 split을 넘는다. 새 split은 `component_id`를 그룹 키로 사용해야 하며, PCM duplicate pair와 같은 원본/재인코딩 계보도 component에 추가 union해야 한다. component 하나를 통째로 train/validation/test 중 하나에 두고 split별 genre·license·길이·codec 분포를 다시 맞춘다.

## source·codec shortcut 위험

FMA real은 전부 MP3이고 sample rate, channel, bitrate, ID3 tag 구조가 다양하다. WaveFake·Echoes fake가 WAV 또는 특정 provider codec으로만 들어오면 모델은 생성 여부 대신 컨테이너·bitrate·대역폭·tag를 학습할 수 있다.

1. 파일명, 경로, ID3, byte size, provider/album field는 feature에서 제거한다.
2. split을 먼저 고정하고 양 클래스에 동일한 decode/resample/channel/loudness/codec augmentation을 적용한다.
3. raw-container 조건과 PCM-normalized 조건을 별도로 평가한다.
4. codec/provider-held-out test를 두고 source 분류 정확도도 함께 측정한다.
5. REAL만 MP3, FAKE만 WAV인 benchmark 결과를 채택하지 않는다.

## 최종 gate

### 확인한 사실

- 공식 archive identity, extracted bytes, 8,000 checksum을 실제로 확인했다.
- MP3/metadata/allowlist ID 조인을 전수 확인했다.
- decode·PCM hash·길이·포맷·무음·클리핑을 8,000개 전부 계산했다.
- strict allowlist 5,130개 중 decodable 5,129개를 확인했다.
- 연결성분과 exact PCM 중복을 split group으로 만들 근거가 생겼다.

### 남은 차단 사유

- 라이선스 URL의 현재 문구·도달성은 기존 메타데이터 감사와 동일하게 미검증이다.
- 1 decode failure, 3 short clips, 1 near-silent allowed clip, 고클리핑 파일의 quarantine 정책을 고정해야 한다.
- WaveFake 안정 archive와 real/fake semantic pairing이 아직 없다.
- Echoes provider별 출력 권리와 manifest ambiguity 감사가 끝나지 않았다.

## 재현 산출물

- `audit_fma_small_audio.py`: 전수 읽기 전용 감사 코드
- `fma-small-audit-run.json`: 기계 판독 집계 SSOT
- `fma-small-audio-inventory.csv`: 8,000개 파일별 SSOT
- `fma-small-duplicate-groups.csv`: decoded PCM 중복 12그룹
- `fma-small-component-summary.csv`: artist-album 연결성분 1,760개
- `fma-small-evidence-map.csv`: 주장-근거-검정 연결표
