DATA_READINESS: READY

# DeepVoice 최종 데이터 준비 판정

2026-08-30 KST 기준으로 실제 보유 원본을 읽기 전용 전수 감사하고, 권리와 기술 조건을 통과한 행만 결합했다. 모델 설계·학습 코드는 이 감사 범위 밖이며 수정하지 않았다.

이 READY는 사용자가 허용한 비영리 해커톤 학습·로컬 검증 범위의 데이터 운영 판정이다. raw corpus 재배포, 상업 이용, 공개 모델 가중치의 라이선스 결론까지 보증하지 않는다. 그 범위가 필요하면 별도 권리 검토가 필요하다.

이진 target은 음원 생성 계보에 따른 `real`/`synthetic`이다. manifest 한 행은 원본 음원 asset 하나이며, 이후 segment를 만들더라도 먼저 정해진 원본 행의 split 안에서만 생성해야 한다.

## 최종 학습 가능 manifest

| 데이터 | label | 학습 가능 행 | 주요 조건 |
|---|---|---:|---|
| LJSpeech 1.1 | real | 13,100 | Public Domain, 13,100 WAV 전수 파싱·메타데이터 join 통과 |
| WaveFake 1.2.0 | synthetic | 117,983 | CC BY-SA 4.0; TTS exact duplicate 사본 16,283개 제거 |
| FMA small | real | 5,129 | strict CC allowlist 5,130 중 decode 실패 133297 제외 |
| AIME open-model 36-shard subset | synthetic | 1,116 | AIME 직접 CC BY 4.0 grant; Suno/Udio 미취득·제외 |
| 합계 | real 18,229 / synthetic 119,099 | **137,328** | 약 **268.412시간** |

최종 manifest는 `deepvoice-training-manifest.csv.gz`다. SHA-256은 `2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6`, 압축 크기는 14,236,928 bytes다. 압축 전 CSV는 104,802,355 bytes, SHA-256 `5261bc9684030a15d6c921780ad27338ca10ad799116c93231d6531a149eaf9d`다. 최종 reports에는 평문 대형 CSV를 배포하지 않는다.

결정적 content-group split의 행 수는 train 110,059, validation 13,540, test 13,729이며, 35,779개 content group 중 둘 이상의 split에 걸친 그룹은 0이다.

할당은 `SHA-256("deepvoice-final-content-group-split-v1|<content_group_key>")`의 앞 64 bit를 `[0,1)`로 바꿔 80%/10%/10% 구간에 넣는다. AIME의 semantic group은 normalized description/prompt 문자열의 exact hash이며 의미 임베딩 기반 근접 군집은 아니다.

## READY의 의미와 제외 범위

READY는 현재 curated manifest로 누수 방지 학습을 시작할 기술·권리 근거가 갖춰졌다는 뜻이다. 모든 원천 데이터가 자동 허용된다는 뜻은 아니다.

READY는 곧바로 “공정한 단일 벤치마크”를 뜻하지도 않는다. 전체 manifest는 real 18,229 대 synthetic 119,099로 불균형하고, FMA는 real-only, AIME·JSUT·TTS는 synthetic-only라 label과 dataset/source가 강하게 결합돼 있다. 전체 행을 섞어 얻은 하나의 점수는 source 식별 성능을 deepfake 탐지 성능으로 오인할 수 있다.

- WaveFake ZIP의 nested TTS copy 16,283개는 canonical row와 file/PCM exact duplicate라 제외했다.
- FMA small 8,000개 중 ND/custom/restricted/unknown 및 비허용 행, 기술 실패를 제외했다. strict allowlist 내 decode 실패 133297도 제외했다.
- AIME에서는 공식 저자의 직접 CC BY 4.0 배포를 적용한 open-model 9종 subset만 쓴다. Suno/Udio는 이 subset에 없다.
- Echoes는 provider별 output-rights 증거가 조건부/HOLD이므로 이 최종 manifest에 포함하지 않았다.
- WaveFake의 Common Voice prompt는 생성용 텍스트 계보일 뿐 upstream reference audio가 ZIP에 포함된 것은 아니다.

과거 `ljspeech-audit-run.json`, `fma-small-audit-run.json`, `aime-subset-audit-run.json`의 `BLOCKED`는 WaveFake 다운로드 전 시점의 단계별 상태다. 현재 판정의 SSOT는 `deepvoice-final-audit-run.json`과 이 문서다.

## 필수 split·평가 규칙

1. random row split과 segment split을 금지한다.
2. LJSpeech real과 WaveFake 7개 vocoder 출력은 동일 LJSpeech ID로 묶는다.
3. JSUT 두 generator는 동일 basic5000 ID로 묶는다.
4. WaveFake TTS는 prompt ID별로 묶고 nested duplicate copy는 읽지 않는다.
5. FMA는 artist-album 이분 그래프 connected component를 indivisible group으로 사용한다. 기존 track split과 충돌하면 component split을 우선한다.
6. AIME는 normalized prompt/description semantic hash를 indivisible group으로 사용한다.
7. content split 성능 외에 generator/provider-held-out, source-family-held-out 결과를 별도 보고한다.
8. sample rate, channels, codec, 길이의 source shortcut을 감시한다. 리샘플링·정규화는 split 확정 뒤 fold 내부에서만 fitting/적용한다.
9. 주 speech 벤치마크는 동일 LJSpeech ID의 real 13,100개와 WaveFake LJS 기반 synthetic 91,700개를 사용하고, ID split 안에서 class-balanced sampling/metric을 적용한다.
10. JSUT·TTS synthetic-only와 FMA/AIME music 조합은 별도 domain stress test로 보고한다. 독립적인 real counterpart가 없는 source-family를 섞은 전체 정확도를 일반화 성능으로 주장하지 않는다.

## 라이선스 운영

- [WaveFake v1.2.0](https://zenodo.org/records/5642694): 저자/DOI/CC BY-SA 4.0 attribution, 변경 표시, adapted material 공유 시 SA.
- [AIME pinned README](https://huggingface.co/datasets/disco-eth/AIME/blob/b84d4be5eda830b6eb714998569dba73530f2601/README.md): revision `b84d4be5eda830b6eb714998569dba73530f2601`과 ETH/AIME attribution, CC BY 4.0 조건 유지.
- [FMA official repository](https://github.com/mdeff/fma): 행별 artist license를 따르므로 각 row의 `license_id`, URL, attribution, NC/SA flag를 manifest에 보존. 사용자는 비영리 CC 자원 사용을 허용했다.
- [LJSpeech official page](https://keithito.com/LJ-Speech-Dataset/): Public Domain 근거와 원 녹음의 MP3 계보 경고를 함께 보존.

WaveFake SA가 학습 가중치에 적용되는지에 관한 법적 판단은 이 기술 감사가 확정하지 않는다. raw audio·가공 corpus를 공유하지 않는 것을 기본으로 하고, 공개 배포 전 별도 검토한다. Common Voice prompt 원문은 최종 manifest에 넣지 않고 ID만 보존한다.

## 산출물과 재현

- 최종 행 SSOT: `deepvoice-training-manifest.csv.gz`
- 집계: `deepvoice-training-manifest-summary.csv`
- 최종 실행 SSOT: `deepvoice-final-audit-run.json`
- split 규칙: `deepvoice-split-constraints.md`
- WaveFake 전수 감사: `wavefake-audit.md`, `wavefake-audit-run.json`
- 재현 코드: `audit_wavefake_stream.py`, `build_deepvoice_training_manifest.py`, `package_large_csv.py`

`build_deepvoice_training_manifest.py`는 `.csv`와 `.csv.gz` WaveFake inventory를 모두 읽을 수 있다. 평문 manifest는 재현 시 임시로 생성하고, 최종 배포에는 `package_large_csv.py`가 만든 결정적 gzip만 둔다.

manifest schema는 dataset/label/sample ID, source family/provider, content group/split, source locator, codec/rate/channels/duration, file·PCM hash, 행별 license/attribution flag, technical/eligibility reason, leakage note를 포함한다. exact duplicate 검사는 file bytes와 decoded/stored PCM을 각각 판정했다. perceptual near-duplicate 전역 검색은 수행하지 않았으므로 source-group 제약과 held-out stress test를 유지한다.
