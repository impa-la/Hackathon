DATA_READINESS: BLOCKED

# FMA 메타데이터·라이선스 감사

- 감사 시각: 2026-08-30, 실행 시각 `2026-08-30T07:01:11Z`
- 원본: `C:\Users\MY PC\Desktop\Hackathon\deepvoice\data\raw\fma\official\fma_metadata.zip`
- 바이트: `358,412,441`
- SHA-256: `d9527a5297a65da31c5676484d5047c3e2b8a8060ce72a46e26158be736bf265`
- 네트워크 요청: 0
- 판정 범위: FMA 메타데이터와 과거 라이선스 URL의 보수적 선별
- 차단 사유: 이 메타데이터 감사 실행 시점에는 실제 FMA 오디오가 없고 `fma_small.zip`은 0바이트였다. 감사 종료 뒤 시작된 다운로드·추출은 이 보고서 수치에 포함하지 않았으며 별도 오디오 감사가 필요하다. 라이선스 URL의 현재 도달 가능성과 약관 변경도 네트워크 없이 확인하지 않았다. 또한 FMA는 실제 음악 원천만 제공하며 대회용 AI 생성 음악·성분 라벨이 없다.

## 1. 원본·추출 무결성

| 검사 | 결과 |
|---|---:|
| ZIP 멤버 | 12개 |
| CRC 실패 | 0 |
| 경로 순회·절대 경로 | 0 |
| 추출 파일 크기와 ZIP 멤버 크기 불일치 | 0 |
| 내장 SHA-1 체크섬 | 10개 중 10개 일치 |
| 감사 중 원본 크기·mtime 변화 | 없음 |

내장 `README.txt`는 이 ZIP이 FMA 음악 분석 데이터셋의 일부라고 설명하고 코드·데이터 URL `https://github.com/mdeff/fma`, 논문 URL `https://arxiv.org/abs/1612.01840`을 기록한다. 각 MP3는 아티스트가 정한 라이선스를 가진다고 적는다.

이 URL은 로컬 메타데이터에 저장된 출처 증거다. 이번 감사는 네트워크 요청을 하지 않았으므로 2026-08-30 현재 페이지 도달 여부, 리다이렉트, 다운로드 가능성, 라이선스 문구 변경을 확인하지 않았다.

세부 결과: `fma-integrity-inventory.csv`, `fma-audit-run.json`

## 2. 트랙·subset·split 구조

`raw_tracks.csv`는 109,727행이고 정제된 `tracks.csv`는 106,574행이다. 정제 트랙 ID는 모두 raw에 있으며 raw에서 정제본으로 들어오지 않은 ID는 3,153개다. 두 표 모두 중복 track ID는 0이다.

### 최소 subset 라벨

| `set.subset` 라벨 | 트랙 수 |
|---|---:|
| small | 8,000 |
| medium | 17,000 |
| large | 81,574 |
| 합계 | 106,574 |

subset은 중첩 bundle이다. small bundle은 small 라벨 8,000개, medium bundle은 small+medium 25,000개, large bundle은 전체 106,574개다.

### 라벨별 split

| 최소 subset 라벨 | training | validation | test | 합계 |
|---|---:|---:|---:|---:|
| small | 6,400 | 800 | 800 | 8,000 |
| medium | 13,522 | 1,705 | 1,773 | 17,000 |
| large | 64,431 | 8,453 | 8,690 | 81,574 |
| 전체 | 84,353 | 10,958 | 11,263 | 106,574 |

medium bundle 누적 split은 19,922/2,505/2,573으로 총 25,000개다.

## 3. 라이선스 allowlist 정의

이번 allowlist는 DeepVoice 대회 규칙의 공개·최소 비영리 사용 조건을 전제로 한 보수적 1차 필터다. 법률 판단이나 현재 라이선스 검증이 아니다.

허용:

- `CC0`
- `CC BY`
- `CC BY-SA`
- `CC BY-NC`
- `CC BY-NC-SA`

제외:

- `EXCLUDE_ND`: CC BY-ND, CC BY-NC-ND
- `EXCLUDE_CUSTOM`: Public Domain Mark·legacy Public Domain을 포함해 엄격 allowlist 밖의 비표준/기타 URL
- `EXCLUDE_RESTRICTED`: FMA-Limited, Music Sharing, Sound Recording Common Law, Orphan Work 표식
- `EXCLUDE_UNKNOWN`: URL 누락 또는 식별 불가

Public Domain Mark나 다른 자유 라이선스가 실제로 사용 불가능하다는 뜻은 아니다. 이번 자동 allowlist가 명시적으로 인정한 5개 유형 밖이므로 별도 검토 대상으로 보류한 것이다.

## 4. 전체 라이선스 분포

| 분류 | 트랙 수 | 비율 |
|---|---:|---:|
| CC BY-NC-SA | 43,512 | 40.828% |
| CC BY-NC | 8,313 | 7.800% |
| CC BY | 6,960 | 6.530% |
| CC BY-SA | 2,802 | 2.629% |
| CC0 | 1,014 | 0.951% |
| ND 제외 | 42,831 | 40.190% |
| custom 제외 | 603 | 0.566% |
| restricted 제외 | 451 | 0.423% |
| unknown 제외 | 88 | 0.083% |

- 전체 허용: **62,601 / 106,574 = 58.739%**
- 전체 제외: **43,973 / 106,574 = 41.261%**

정확한 라이선스 title·URL 조합별 분포는 `fma-license-summary.csv`의 `license_detail` 행에 있다.

## 5. small subset 라이선스 분포

| 분류 | 트랙 수 |
|---|---:|
| CC BY-NC-SA | 2,884 |
| CC BY-NC | 930 |
| CC BY | 893 |
| CC BY-SA | 369 |
| CC0 | 54 |
| ND 제외 | 2,789 |
| custom 제외 | 75 |
| restricted 제외 | 1 |
| unknown 제외 | 5 |

- small 허용: **5,130 / 8,000 = 64.125%**
- small 제외: **2,870 / 8,000 = 35.875%**

`fma-license-allowlist.csv`는 전체 curated 허용 트랙 62,601행을 담고 있으며 `small_member`로 small 허용 5,130개를 바로 필터링할 수 있다.

## 6. 결측·중복·미발견 표식

### 핵심 결측

| 필드 | 범위 | 결측 |
|---|---:|---:|
| curated subset/split | 106,574 | 0/0 |
| curated artist_id/album_id | 106,574 | 0/0 |
| curated duration | 106,574 | 0 |
| curated license title | 106,574 | 87 |
| curated top genre | 106,574 | 56,976 (53.461%) |
| raw license URL | 109,727 | 88 |
| raw track file path | 109,727 | 7 |
| raw track URL | 109,727 | 1,041 |

전체 열별 결측·고유값·dtype는 `fma-metadata-missingness.csv`에 있다.

### `not_found.pickle`

피클은 임의 클래스 로드를 금지하고 FMA 파일에 실제로 사용된 NumPy scalar/dtype 두 생성자만 허용한 restricted unpickler로 읽었다.

| 표식 | 수 |
|---|---:|
| tracks | 45,594 |
| albums | 480 |
| artists | 250 |
| audio | 180 |
| clips | 286 |

`not_found.tracks` 45,594개는 raw track ID와 겹침이 0이다. raw 109,727개와 합치면 0~155,320 ID를 빠짐없이 덮으므로 원천 크롤링에서 미발견·삭제된 트랙 ID 공백으로 해석할 수 있다. `audio` 180개와 `clips` 286개 ID는 raw에는 모두 있으나 curated에는 0개다. 즉 정제본은 이 오디오·클립 미발견 표식을 배제한다.

### 파일 경로 중복

| 범위 | 중복 경로 그룹 | 영향 트랙 | 초과 행 | 최대 그룹 |
|---|---:|---:|---:|---:|
| raw | 633 | 1,546 | 913 | 18 |
| curated | 609 | 1,481 | 872 | 18 |
| small | 0 | 0 | 0 | 1 |

실제 오디오가 없으므로 같은 경로가 같은 바이트인지, 링크·메타데이터 오류인지, 중복 콘텐츠인지 확인하지 못했다. 전체/medium 사용 시 경로 중복을 원본 그룹으로 먼저 묶어야 한다.

## 7. artist·album split 누수

### Artist

| 범위 | 고유 artist | split 교차 artist | 영향 트랙 |
|---|---:|---:|---:|
| 전체 | 16,341 | 0 | 0 |
| medium bundle | 5,545 | 0 | 0 |
| small | 2,309 | 0 | 0 |

공식 split은 artist ID 수준에서는 완전히 분리돼 있다.

### Album

| 범위 | 고유 album | split 교차 album | 영향 트랙 | train↔val | train↔test | val↔test | 세 split 모두 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 | 14,854 | 671 | 13,997 | 454 | 464 | 253 | 250 |
| medium bundle | 5,152 | 274 | 2,485 | 184 | 175 | 93 | 89 |
| small | 2,464 | 90 | 549 | 58 | 49 | 21 | 19 |

artist-disjoint라고 해서 album-disjoint인 것은 아니다. 컴필레이션·다중 아티스트·메타데이터 관계 때문에 같은 album ID가 split을 넘는다. 음악 원본·녹음 세션·마스터링 흔적 누수를 막으려면 공식 split을 그대로 쓰지 말고 최소한 album과 artist의 연결 성분을 하나의 그룹으로 재분할해야 한다.

## 8. DeepVoice 용도 한계

- FMA는 실제 음악 원천 후보이며 AI 생성 음악 FAKE를 제공하지 않는다.
- 대회의 `MUSIC_PRESENT`, `VOICE_PRESENT`, `MUSIC_FAKE`, `VOICE_FAKE`, `FILE_FAKE` 5개 라벨로 바로 변환할 수 없다.
- 보컬 유무가 신뢰 가능한 구조화 타깃으로 보장되지 않는다. `genre_top`도 53.5% 결측이며 보컬 여부의 대리 라벨로 사용하면 안 된다.
- MP3 오디오를 확보하더라도 코덱 흔적이 REAL만 식별하는 지름길이 되지 않도록 FAKE와 포맷·비트레이트·후처리를 교차 균형화해야 한다.

## 9. 검증 제약과 금지 사항

1. 엄격 allowlist 파일에 있는 ID만 자동 후보로 인정하고, custom/restricted/unknown은 수동 검토 전 사용하지 않는다.
2. 라이선스 URL의 현재 문구와 원천 페이지를 네트워크가 허용된 별도 단계에서 파일별·버전별로 재확인한다.
3. artist ID와 album ID를 함께 묶은 연결 성분으로 분할하고, 같은 track_file·재인코딩·세그먼트를 한 폴드에 둔다.
4. 공식 FMA split을 album-safe split으로 오인하지 않는다.
5. FMA를 REAL 음악으로만 사용한다면 FAKE 음악 소스와 수집·코덱·길이·음량·무음 조건을 교차 균형화한다.
6. 실제 오디오 전수 해시·디코딩·길이·샘플레이트·채널·무음·클리핑 감사 전 학습에 사용하지 않는다.

## 10. 최종 게이트

### 확인한 사실

- 공식 메타데이터 ZIP과 추출 파일은 CRC·SHA-1·크기가 모두 일치한다.
- curated 106,574개, small 8,000개와 split 구조를 실제 파일에서 확인했다.
- 엄격 allowlist는 전체 62,601개, small 5,130개다.
- artist split 겹침은 0이지만 album 겹침은 전체 671개, small 90개다.
- 미발견·결측·경로 중복을 실제 수치로 확인했다.

### 미확인 사항

- FMA 소스·라이선스 URL의 2026-08-30 현재 도달성과 문구
- 실제 FMA MP3 파일의 존재·해시·디코딩·오디오 품질
- 보컬·악기 성분과 대회 5개 라벨
- 실제 오디오 바이트 기준 중복·파생 계보

### 치명적 위험

- 과거 메타데이터 URL만으로 현재 사용권을 확정할 수 없다.
- album 교차 split을 무시하면 동일 앨범 원천 누수가 생긴다.
- FMA REAL MP3와 다른 출처 FAKE 오디오를 그대로 섞으면 코덱·출처 대리변수 누수가 생긴다.

### 다음 단계

1. 후속으로 시작된 FMA small 다운로드·추출이 완전히 끝난 뒤 아카이브 안정성과 8,000개 MP3를 별도 감사한다.
2. allowlist 5,130개 ID와 실제 MP3를 조인해 누락·손상·중복·품질을 전수 감사한다.
3. 라이선스·원천 URL을 현재 웹에서 재검증하고 변경·삭제·추가 약관을 기록한다.
4. album+artist 연결 성분 기준 새 분할을 생성하고 FAKE 음악 소스와 교란 균형을 검증한다.

이 메타데이터 감사만으로는 모델링을 시작할 수 없다.

이 감사 snapshot의 집계 SSOT는 `fma-audit-run.json`이며, 트랙별 허용 근거의 SSOT는 `fma-license-allowlist.csv`다. 이 Markdown의 숫자는 사람이 읽기 위한 요약이다.
