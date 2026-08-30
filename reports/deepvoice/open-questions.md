# DeepVoice 라이선스 미해결 질문·서면 확인 템플릿

확인 기준일: **2026-08-30**. 아래 답변이 오기 전 관련 자산은 다운로드·학습에 넣지 않는다. 답변은 원문 이메일/공식 게시글/PDF로 저장하고 발신자·날짜·범위·첨부파일 hash를 `source-register`에 연결한다.

## 우선순위 P0 — DACON

수신 후보: DACON 대회 문의 게시판. 근거: [공식 규정](https://dacon.io/competitions/official/236749/overview/rules), [실제 파일 전부 제출 답변](https://dacon.io/competitions/official/236749/talkboard/417198?dtype=recent&page=1) (확인 2026-08-30).

1. 2차평가로 제출한 원본·파생·생성 학습 음원은 어떤 사람/기관이 접근하며, 비공개 심사 외 재이용·제3자 제공·향후 공개가 가능한가?
2. 보존기간과 평가 종료 후 삭제 정책은 무엇인가?
3. ShareAlike 자료와 notice/license 파일을 함께 제출하고 심사 접근자에게 그 조건을 고지할 수 있는가?
4. 모델 가중치도 학습 재현에 필요할 경우 제출 대상인지, 공개 download URL+hash로 대체 가능한지?

확인 완료: DACON은 공개 자원이고 최소 비영리 사용이 허용되며 조건을 지키는 경우 CC BY-NC-SA 자료도 사용할 수 있다고 [공식 답변](https://dacon.io/competitions/official/236749/talkboard/417212)했다. 따라서 NC 자체는 더 이상 미해결 질문이 아니다.

권장 질문 문구:

> 본 대회 2차평가의 “학습에 사용한 모든 데이터 파일” 제출과 관련해, 제출 파일의 접근자, 이용 목적, 제3자 제공/공개 여부, 보존기간, 평가 종료 후 삭제 여부를 서면으로 확인 부탁드립니다. CC BY-SA/BY-NC-SA 또는 비영리 연구 허용 자료의 권리자에게 정확한 전달 범위를 설명해야 합니다.

## 우선순위 P0 — ASVspoof 2021/2019

수신 후보: ASVspoof organisers/EURECOM. 근거: [ASVspoof 2021](https://www.asvspoof.org/index2021.html), [ODC-By](https://opendatacommons.org/licenses/by/1-0/) (확인 2026-08-30).

1. 2021 DF의 ODC-By 표시는 database compilation뿐 아니라 **각 FLAC sound recording의 copyright/performance rights**도 포함하는가?
2. 참가자가 DF 원본과 segment/re-encode/augmentation/mix 파생 파일을 classifier 학습에 쓰고, DACON 비공개 심사위원에게 실제 파일 전부를 전달해도 되는가?
3. ASVspoof 2019, VCTK, VCC2018/2020 등 upstream별 notice·license·attribution의 완전한 목록은 무엇인가?
4. judge transfer가 ODC-By의 Convey에 해당한다면 제출 패키지에 넣어야 할 정확한 notice 문구는 무엇인가?
5. 답변은 2021 DF evaluation set 전체 버전/DOI에 적용되는가?

> We plan to use ASVspoof 2021 DF to train a detector for a noncommercial-purpose DACON research competition. Finalists must privately deliver every original and derived training audio file to the judges; metadata or URLs cannot substitute for files. Does the ODC-By grant cover copyright/performance rights in each FLAC, and may we deliver the full files plus segments/re-encodings/mixes to DACON judges? Please specify all upstream notices and licenses.

## 참고 확인 P2 — WaveFake 권장 notice

수신 후보: WaveFake authors. 근거: [Zenodo](https://zenodo.org/records/5642694), [datasheet](https://zenodo.org/record/5642694/files/datasheet.pdf) (확인 2026-08-30).

1. 권장 attribution text와 JSUT/LJSpeech upstream notice는 무엇인가?
2. WaveFake와 CC0/CC BY/BY-SA real audio를 overlay한 adaptation을 CC BY-SA 4.0으로 제출할 때 권장 notice bundle이 있는가?

> We will use the canonical WaveFake Zenodo release under CC BY-SA 4.0 for a noncommercial-purpose DACON detector competition. Please advise the preferred WaveFake, JSUT, and LJSpeech attribution/notice bundle for original and modified files delivered to the judges.

## 우선순위 P0 — MedleyDB

수신 후보: MedleyDB/NYU contact. 근거: [official download terms](https://medleydb.weebly.com/downloads.html) (확인 2026-08-30).

1. 사이트의 “do not republish full or part without consent” 요청 아래, finalists가 original multitracks/stems와 derived segments/mixes를 DACON judges에게 비공개 전달할 수 있는가?
2. 생성 fake music의 conditioning 또는 augmentation source로 써도 되는가?
3. artist별 추가 consent/attribution/notice가 있는가?
4. 심사 종료 후 DACON 삭제 조건을 요구하는가?

> We request explicit consent to use selected MedleyDB tracks in a noncommercial-purpose DACON detector competition and to privately deliver every original and derived training file to the judges. The files will not be published by us. Please state whether this is permitted and list any artist-specific attribution, ShareAlike, retention, or deletion conditions.

## 우선순위 P0 — Meta MusicGen

수신 후보: Meta AudioCraft/MusicGen maintainers or licensing contact. 근거: [weights license](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights), [model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md) (확인 2026-08-30).

1. 생성 WAV의 소유권·적용 라이선스·재배포 조건은 무엇인가? weights의 CC BY-NC가 outputs에 적용되는가, 적용되지 않는가?
2. 생성 WAV를 detector training data로 쓰고 DACON judges에게 실제 파일로 전달해도 되는가?
3. text-only original prompts를 쓸 때 필요한 notice가 무엇인가?
4. weights 자체가 재현 패키지에 포함될 경우 필요한 attribution/source/changes 파일은 무엇인가?

> The model card and repository clearly license code and weights separately, but we could not find an explicit output license. Please confirm the ownership/license of MusicGen-generated WAV files and whether they may be used as training data and privately delivered to judges in a noncommercial-purpose DACON competition.

## 우선순위 P0 — FakeMusicCaps authors

수신 후보: Politecnico di Milano ISPL/FakeMusicCaps authors. 근거: [Zenodo v2](https://zenodo.org/records/15063698), [GitHub license](https://github.com/polimi-ispl/FakeMusicCaps/blob/main/LICENSE), [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295441/) (확인 2026-08-30).

1. Zenodo v2 audio ZIP에 적용되는 정확한 license는 무엇인가? GitHub repository의 CC BY-NC 4.0이 audio ZIP 전체에 적용되는가?
2. 5개 generator별로 생성 output 재배포 권리를 어떤 근거로 확보했는가?
3. MusicCaps captions의 사용·재배포 허락은 무엇인가?
4. 비영리 목적 DACON 학습과 judges에게 27,605 audio 또는 사용 subset 실제 파일 전달이 허용되는가?
5. 파일별 generator ID, prompt, source caption, model version, license mapping을 제공할 수 있는가?

> The Zenodo v2 record currently shows no Rights/License field, while the GitHub repository contains a CC BY-NC license. Please confirm in writing whether that license covers every audio file in the Zenodo ZIP and provide model-by-model output and caption rights permitting private delivery of the actual files to DACON judges.

## 우선순위 P1 — LibriSpeech/LibriVox

근거: [OpenSLR](https://www.openslr.org/12), [LibriVox public domain policy](https://librivox.org/pages/public-domain/) (확인 2026-08-30).

1. OpenSLR의 CC BY 4.0 grant가 개별 sound recording과 transcription 모두에 적용되는가?
2. 한국에서 원작/번역 copyright 상태를 검증할 수 있는 title/work metadata가 있는가?
3. DACON judges에게 original/derived clips 전달 시 권장 attribution은 무엇인가?
4. 화자 동의·personality/privacy 관련 별도 제한이 있는가?

답변이 없으면 한국에서도 사후 70년 등 만료가 명백한 원작·번역 title만 allowlist하고 저자/번역자 사망연도 근거를 저장한다.

## 우선순위 P1 — Stability AI / Stable Audio Open Small

근거: [model page](https://huggingface.co/stabilityai/stable-audio-open-small), [Community License](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE), [AUP](https://stability.ai/use-policy) (확인 2026-08-30).

1. 생성 audio를 classifier training에 쓰는 것이 “create or improve any foundational generative AI model” 금지와 무관함을 확인할 수 있는가?
2. user-owned output을 DACON judges에게 실제 파일로 전달해도 되는가?
3. outputs만 전달할 때도 Notice/“Powered by Stability AI” 조항이 적용되는가, 아니면 Stability Materials/Derivative Works 배포에만 적용되는가?
4. 어떤 entrant/account가 gated terms에 동의해야 하는가?

## 우선순위 P1 — FMA subset 운영 질문

근거: [FMA repository](https://github.com/mdeff/fma), [FMA paper](https://arxiv.org/abs/1612.01840) (확인 2026-08-30).

1. archive `tracks.csv`의 license와 현재 FMA track page가 다르면 어느 시점을 기준으로 해야 하는가?
2. deleted/unreachable track은 archive metadata만으로 재배포 가능한가, 아니면 제외해야 하는가?
3. artist-chosen CC license가 composition, performance, sound recording 모두를 포함한다고 보증하는가?
4. DACON file transfer를 위해 제공되는 권장 per-track attribution schema가 있는가?

답변이 없으면 source page가 현재 접근되고 CC0/BY/BY-SA가 명시된 track만 사용하며, ND/NC/NC-SA/custom/restricted/unknown과 URL 소실 track은 제외한다.

## 내부 결정이 필요한 질문

1. 참가 주체가 개인인지 팀/기관인지, 연매출·상업 활동과 모델 access account의 명의는 누구인지?
2. 상금 수령과 향후 기술의 상업 활용을 분리할 수 있는가? NC 자산으로 학습한 model/checkpoint를 대회 이후 상업 프로젝트에 재사용하지 않을 수 있는가?
3. license stratum별 별도 model을 학습할지, 하나의 model에 여러 라이선스 원본을 넣을지?
4. BY-SA/BY-NC-SA 원본으로 만든 augmentation을 심사위원에게 어떤 license bundle로 전달할지?
5. DACON이 제출 파일을 삭제하지 않거나 재사용한다면 해당 자산을 포기할지?
6. rights holder 회신이 마감 전에 오지 않으면 안전 조합(ASVspoof 2015 + 검증된 LibriSpeech/FMA subset + Stable Audio Open Small 확인분)만으로 축소할지?

## 승인 gate

다음 중 하나라도 `unknown`이면 데이터 취득 승인을 내리지 않는다.

- 실제 음원 asset에 적용되는 명시 license
- 최소 비영리 사용 허용(DACON 대회 적격성)
- segmentation/re-encoding/augmentation/mixing 허용
- DACON judges에게 실제 파일 전달 허용
- required attribution/SA/notice/source 제공 가능
- 제3자 원작·화자·artist 권리 확인
- access terms 동의 주체와 termination/deletion 처리 가능
- 서로 혼합될 asset의 ShareAlike compatibility
