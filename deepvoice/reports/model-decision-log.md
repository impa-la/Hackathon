# DeepVoice 모델 결정 로그

결정일: 2026-08-30
상태: 설계 결정 완료, 구현·학습 미수행
데이터 게이트: `DATA_READINESS: READY`

## D01. frozen dual-specialist가 핵심안이다

**결정:** speech에는 WavLM Base Plus, music에는 MERT-v1-95M을 frozen encoder로 사용하고, 저수준 artifact branch와 task-specific soft gate로 결합한다.

**근거:**

- 전체 137,328행 중 speech 축은 LJSpeech real 13,100과 WaveFake synthetic 117,983이 지배한다.
- music 축은 FMA real 5,129와 AIME synthetic 1,116으로 작고, speech와 표현·codec·대역이 다르다.
- 하나의 scratch 대형 모델을 268.412시간과 L4 한 장에서 학습하는 것은 비효율적이다.
- 95M급 encoder 두 개는 300M~1B 통합 모델보다 package와 runtime을 예측하기 쉽다.
- frozen first는 source shortcut 과적합과 derived-weight 권리 문제를 줄인다.

**검증 조건:** E03/E04 single branch가 각 domain에서 baseline을 이기고, E05 fusion이 best branch보다 RobustSelectionScore 0.005 이상 개선해야 한다.

## D02. metadata 없는 canonical audio 입력

**결정:** 파일명, 경로, generator/provider, codec tag, 원 sample rate/channel metadata를 feature로 사용하지 않는다. WavLM과 artifact primary view는 canonical 16 kHz, MERT는 이 16 kHz view를 24 kHz로 재변환한다.

**근거:**

- WaveFake generator/sample-rate, AIME provider/prompt, FMA codec/artist-album이 label과 결합되어 있다.
- official test는 16 kHz이며 mono/stereo, MP3/WAV/FLAC와 전화음이 혼재한다.
- FMA의 원 44.1 kHz를 MERT에 직접 주고 speech만 upsample하면 source bandwidth가 label proxy가 된다.

**조건부 검토:** 모든 source에 같은 규칙을 적용하는 fixed 32 kHz LFCC/phase view를 E06H에서 시험한다. source/rate/codec probe와 official 16 kHz stress를 통과할 때만 채택한다.

## D03. artifact branch는 LFCC/phase가 기본, CQT는 ablation이다

**결정:** LFCC 또는 log-linear filterbank, delta/delta2, phase 및 instantaneous-frequency residual을 작은 LCNN/ResNet으로 처리한다. CQT는 music artifact에 대한 E06 ablation이다.

**근거:**

- WaveFake는 7개 LJS generator와 JSUT/TTS 계열이 있어 vocoder·phase 흔적을 semantic encoder와 독립적으로 볼 필요가 있다.
- codec/sample-rate shortcut도 artifact branch가 가장 쉽게 학습할 수 있으므로 shortcut probe와 대칭 augmentation을 필수로 묶는다.
- CQT는 music harmonic 구조에는 유리하지만 계산량이 늘고 codec fingerprint에 과적합할 수 있다.

## D04. five-head mask가 label의 의미를 보존한다

**결정:** 각 source가 실제로 증명하는 label만 BCE에 포함한다. 알 수 없는 voice/music fake label은 negative로 채우지 않는다.

**근거:**

- LJS/WaveFake는 voice 진위가 관측되지만 music 진위는 정의되지 않는다.
- FMA는 real music이지만 voice absence를 증명하지 않는다.
- AIME는 공식 논문상 10초 instrumental-only이므로 voice absence와 music fake를 관측할 수 있다.
- 현재 manifest는 single-domain뿐이고 BOTH_PRESENT/NEITHER_PRESENT가 없어 joint presence의 완전한 식별이 불가능하다.

**파생 결정:** direct FILE_FAKE와 `presence × conditional fake` 논리식을 differentiable consistency/CPS 보조 loss로 묶되, 필요한 label이 모두 관측된 row에서만 적용한다.

## D05. manifest split을 절대 바꾸지 않는다

**결정:** `recommended_content_split`을 고정하고 content, generator/provider-held-out, source-family-held-out을 동시에 보고한다.

**근거:**

- LJSpeech와 7개 WaveFake generator는 같은 content ID의 1:1:7 pairing이다.
- AIME는 332 unique description, 270 duplicate group, 1,054 duplicate row가 있어 prompt leakage 위험이 크다.
- FMA는 artist-album 연결 성분이 있고 random row split은 artist/album leakage를 만든다.
- 현재 최종 manifest는 35,779 content group과 split 교차 0개를 이미 검증했다.

## D06. sampling은 source count가 아니라 task balance를 맞춘다

**결정:** speech/music 50:50, 각 domain real/fake 50:50을 목표로 하고 content group을 먼저 뽑는다. LJS fake generator와 AIME provider를 균등 회전한다.

**근거:**

- raw count는 synthetic 119,099 대 real 18,229로 심하게 불균형하다.
- LJS의 동일 content에 fake 7개가 있어 행 단위 sampling은 특정 문장과 fake가 gradient를 지배한다.
- AIME는 provider당 124개로 균일하지만 prompt 중복을 group으로 다뤄야 한다.

## D07. mixed presence는 조건부 새 데이터다

**결정:** 현재 설계의 core training에 임의 mixture를 넣지 않는다. BOTH_PRESENT/NEITHER_PRESENT는 E10과 새 data audit를 통과한 뒤에만 사용한다.

**근거:**

- 현재 실제 혼합 자료가 없다.
- waveform mix는 단순 augmentation이 아니라 결합 저작물과 새 lineage를 만든다.
- 기존 split에 섞으면 양쪽 content group의 leakage가 발생할 수 있다.

**권리 규칙:** CC0, CC BY, 또는 서로 호환되는 CC BY-SA stratum만 사용한다. NC와 SA 등 조건이 충돌하거나 불명확한 waveform 조합을 금지한다.

## D08. WavLM·MERT·head를 별도 artifact로 제출한다

**결정:** WavLM 원 checkpoint, MERT 원 checkpoint, trainable adapter/head를 한 opaque tensor 파일로 병합하지 않는다. 각각 exact hash와 notice를 유지한다.

**근거:**

- WavLM weight는 공식 model card가 CC BY-SA 3.0 license로 연결하고, unilm code MIT와 다르다.
- MERT weight는 CC BY-NC 4.0이며 attribution·license·변경 고지가 필요하다.
- 별도 파일은 원본/변경물 경계를 보여주고 branch 교체 및 권리 fallback을 가능하게 한다.

**MERT 추가 조건:** exact revision의 custom code를 vendor하고 runtime `trust_remote_code`와 network를 금지한다.

## D09. XLS-R는 권리 fallback이다

**결정:** `facebook/wav2vec2-xls-r-300m`을 core가 아니라 Apache-2.0 fallback으로 유지한다.

**근거:**

- weight/model card가 Apache-2.0으로 명확하다.
- 300M급, 약 1.27GB weight라 WavLM/MERT base 각각보다 느리고 크다.
- speech representation이므로 MERT music specialization을 그대로 대체하지 못한다.

**승격 조건:** core와 score 차이 0.01 이하, 50분 이하, peak GPU 18 GiB 이하.

## D10. BEATs는 유망하지만 조건부다

**결정:** unified encoder의 최우선 후보지만 exact checkpoint 획득·hash·weight 권리를 확인할 때까지 `BLOCKED_RIGHTS`다.

**근거:**

- speech/music을 한 encoder로 처리해 runtime을 줄일 가능성이 있다.
- official unilm repo code는 MIT이나 checkpoint가 별도 OneDrive 배포이며 exact file hash와 별도 weight 재배포 근거가 현재 package에 고정되지 않았다.

## D11. PANNs와 HTS-AT는 core에서 기각한다

**결정:** 현재는 routing이나 fallback에도 넣지 않는다.

**근거:**

- PANNs repo code는 MIT이고 bundled Cnn14 checkpoint hash도 확인했지만 Zenodo record의 license field가 비어 있어 weight 재배포 grant가 불명확하다.
- HTS-AT repo code는 MIT이나 Google Drive checkpoint의 exact hash와 weight 권리가 불명확하다.

**재검토 조건:** 공식 weight license/terms, exact URL/revision/hash, DACON model-file submission 재배포 가능성을 모두 문서화한다.

## D12. DF Arena 1B와 htdemucs는 기각한다

**DF Arena 1B:** 4.591GB bundled weight, 1.147B급, HF license가 `other`라 10GB package와 60분 L4 및 권리 면에서 core보다 불리하다.

**htdemucs:** source separation은 music/voice routing에 매력적이지만 weight 권리가 code MIT와 별도로 명확하지 않고, 별도 separator pass가 runtime을 크게 늘린다.

## D13. calibration은 정적이고 파일 독립적이다

**결정:** validation-only head temperature, segment aggregation blend, FILE_FAKE alpha만 사용한다. provider calibrator와 test-batch prior 보정은 사용하지 않는다.

**근거:**

- EER/AUC는 단조 temperature만으로 개선되지 않는다.
- calibration의 실익은 CPS, Brier, direct/logic 정합성이다.
- test batch 통계는 파일 독립 inference 제약을 위반한다.

## D14. 추론 cap은 8초 × 8 segment다

**결정:** 파일당 최대 8개 deterministic segment로 dual encoder를 실행한다.

**근거:**

- 60초 파일 전체를 sliding-window로 훑으면 1,200파일/60분 budget이 불안정하다.
- 최대 9,600 segment와 19,200 encoder pass로 예산을 미리 계산할 수 있다.
- 시작/끝 포함 균등 위치는 단일 crop보다 국소 fake 흔적을 포착한다.

**fallback:** cap 6은 score 감소 0.005 이하에서만 허용한다.

## 변경 통제

다음 변경은 이 로그와 관련 CSV를 함께 갱신해야 한다.

- checkpoint 이름, revision, hash, license
- split/group/sampling 또는 label mask
- segment 길이/cap과 inference budget
- mixed data 또는 새로운 외부 데이터
- encoder fine-tune/LoRA와 derived checkpoint 재배포 판단
- test를 여는 시점이나 calibration 자료
