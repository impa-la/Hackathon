# DeepVoice 외부 데이터 준비도

- 기준일: 2026-08-30
- 결론: `DATA_READINESS: READY`
- 모델 설계 게이트: 열림. 최종 SSOT는 `deepvoice-final-data-readiness.md`와 `deepvoice-final-audit-run.json`이다.

## 현재 사용할 수 있는 축

| Domain | Real | Fake | 현재 판단 |
|---|---|---|---|
| Speech | LJSpeech 13,100개, 23.9214시간 | WaveFake 1.2.0 고유 생성 117,983개 | real/fake 전수 감사 및 ID-group 결합 완료 |
| Music | FMA small 중 license strict allowlist 5,130개, 정상 decode 5,129개 | AIME 공개 모델 subset 1,116개; Echoes | AIME 전수 감사·권리 판정 완료, Echoes 4,462개는 권리 계보 HOLD |

LJSpeech는 전수 파싱·metadata 1:1·형식 검증을 통과했다. 동일 normalized text 10그룹 36개는 반드시 같은 split에 묶는다. FMA small은 strict allowlist를 적용하고, decode 오류·PCM 중복·album/artist connected component와 clipping 품질 플래그를 반영한 group split이 필요하다.

Echoes는 archive SHA-256·CRC·경로 안전성에는 문제가 없지만 실제 ZIP에 bona-fide 오디오가 없다. manifest의 유효 4,462개 generated audio는 dataset-level CC BY-SA 4.0 표시만으로 각 생성 제공자의 출력물 이용권을 입증할 수 없다. exact checkpoint/API revision, 생성 당시 account tier, provider별 output terms와 FMA reference 연결 증거가 보강되기 전까지 학습에서 제외한다.

AIME는 pinned revision의 210개 shard, 6,500행과 제공자별 500개 구성을 projected metadata audit으로 확인했다. 저자가 생성한 6,000개 트랙을 CC BY 4.0으로 배포한다고 공식 dataset card에 명시하고, 논문도 12개 생성 모델별 500개와 공개 배포를 설명한다. 전체 62.28GB 대신 Suno·Udio를 제외하고 MusicGen 3종, AudioLDM 2종, Riffusion, Mustango, Stable Audio 2종에서 pure-provider shard 4개씩을 골라 1,116행·1.246GB를 취득했다. 36/36 파일이 official LFS SHA-256·크기·행 수와 일치했고 1,116/1,116 오디오가 전부 decode됐다. exact byte/PCM 중복과 nonfinite sample은 0이다. 이 1,116행은 AIME attribution을 유지하는 비영리 학습에 `training_eligible=true`이며, Suno·Udio와 per-track license 미검토 MTG-Jamendo는 제외한다. 상세 판정은 `aime-audit.md`와 `aime-rights-matrix.csv`를 따른다.

## 완료된 게이트

1. WaveFake 공식 archive의 28,918,626,084 bytes와 MD5 `76b3e62d69f866e57ad6b1debaff434b`를 확인하고 local SHA-256을 기록했다.
2. ZIP 134,266개를 영구 해제 없이 전수 스트리밍 검사해 CRC·WAV 파싱 실패 0을 확인했다. 고유 생성음 117,983개와 중복 사본 16,283개를 분리했다.
3. LJSpeech·WaveFake·FMA·AIME를 source ID, artist-album component, prompt group 단위로 결합해 split 교차 그룹 0인 매니페스트를 만들었다.
4. 최종 `deepvoice-training-manifest.csv.gz` 137,328행에 license, source version, sample hash, training eligibility와 제외 이유를 고정했다.
5. 모델 설계는 이 데이터 계약과 `deepvoice-split-constraints.md`를 필수 입력으로 사용한다.
