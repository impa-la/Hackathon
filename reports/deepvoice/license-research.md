# DeepVoice 외부 학습자원 라이선스·이용조건 실무 감사

- 확인 기준일: **2026-08-30**
- 범위: DACON DeepVoice 대회용 외부 학습 데이터, 생성 모델, 코드, 가중치, 생성 출력물
- 성격: **법률 자문이 아니라 실제 참가·2차평가 제출을 위한 실무 라이선스 리스크 감사**이다. 권리자·DACON의 서면 확인과 필요시 전문 법률 검토를 대체하지 않는다.
- 제한: 외부 파일·모델은 다운로드하지 않았고, 학습/모델링 코드도 수정하지 않았다.

## 결론부터

현재 문서만으로 바로 사용할 수 있는 핵심 축은 **ASVspoof 2015(CC BY 4.0)**, **LJSpeech(public domain)**, **WaveFake(CC BY-SA 4.0)**이다. 실화자 음성 보강용 **LibriSpeech(CC BY 4.0)**는 한국에서의 LibriVox 원작·녹음 권리 범위를 제목 단위로 점검하는 조건을 붙인다. WaveFake는 Zenodo의 CC BY-SA와 datasheet의 JSUT 권리자 연구목적 배포 허락을 근거로 GO로 올리되, 귀속·동일조건·변경 표시를 지키고 BY-SA 호환 계층 안에서만 합성한다. [ASVspoof 2015 DataShare](https://datashare.ed.ac.uk/items/a714c9b8-4acd-43a2-ac45-68eebcd070de/full), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/), [LibriSpeech OpenSLR](https://www.openslr.org/12), [WaveFake Zenodo](https://zenodo.org/records/5642694), [WaveFake datasheet](https://zenodo.org/record/5642694/files/datasheet.pdf) (모두 확인 2026-08-30).

기존에 유력해 보였던 **ASVspoof 2021 DF + MedleyDB + MusicGen** 조합은 곧바로 GO가 아니다. ASVspoof 2021의 ODC-By는 데이터베이스와 개별 음원 권리를 분리하고, MedleyDB는 CC BY-NC-SA 외에 사이트가 별도로 재게시 자제를 요청하며, MusicGen은 가중치만 CC BY-NC이고 출력물 라이선스를 명시하지 않는다. DACON이 학습에 사용한 실제 파일 전부를 요구하므로 세 후보 모두 권리자 확인 전에는 **CONDITIONAL**이다. [ASVspoof 2021](https://www.asvspoof.org/index2021.html), [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/), [MedleyDB download terms](https://medleydb.weebly.com/downloads.html), [AudioCraft repository](https://github.com/facebookresearch/audiocraft), [MusicGen model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md) (확인 2026-08-30).

**MUSDB18 전체와 FakeMusicCaps v2는 현 상태에서 NO-GO**다. MUSDB18은 다수 트랙이 Restricted/academic-only 계보이고, FakeMusicCaps의 Zenodo 음원 ZIP에는 명시된 Rights/License가 없으며 GitHub의 CC BY-NC 파일이 별도 Zenodo 데이터 ZIP에 적용되는지 불명확하다. [MUSDB18 official page](https://sigsep.github.io/datasets/musdb.html), [MUSDB18 track list](https://raw.githubusercontent.com/sigsep/website/master/content/datasets/assets/tracklist.csv), [FakeMusicCaps Zenodo](https://zenodo.org/records/15063698), [FakeMusicCaps GitHub license](https://github.com/polimi-ispl/FakeMusicCaps/blob/main/LICENSE) (확인 2026-08-30).

## 1. DACON이 만든 핵심 제약

DACON 규정은 공개 자원이면서 최소 비영리 목적 사용이 허용된 데이터·모델·API 등을 허용하되, 참가자가 라이선스와 이용조건을 직접 확인하도록 한다. 또한 2차평가 대상자는 출처를 밝히고, 재현 가능한 학습 코드·모델 보고서·학습 데이터 보고서와 함께 **학습에 사용한 모든 데이터 파일**을 제출해야 한다. [DACON 규정](https://dacon.io/competitions/official/236749/overview/rules) (확인 2026-08-30).

운영자의 공식 답변은 생성 음원을 포함한 실제 학습 파일 전부가 필요하고 메타데이터로 대체할 수 없으며, 대용량이면 Google Drive 등으로 전달해도 된다고 재확인했다. 따라서 “공개 URL만 제출” 또는 “생성 seed만 제출”은 불충분하다. [DACON 공식 답변](https://dacon.io/competitions/official/236749/talkboard/417198?dtype=recent&page=1) (확인 2026-08-30).

대회는 별도 train 데이터를 제공하지 않고 참가자가 구성한다. 공개 1,200개 파일은 public test이고 같은 수량의 private test가 별도 존재한다. [DACON 데이터 페이지](https://dacon.io/competitions/official/236749/data), [대회 개요](https://dacon.io/competitions/official/236749/overview/description) (확인 2026-08-30).

실무상 심사위원에게 원본 또는 변형 음원을 전달하는 행위를 라이선스의 “공유/배포/Convey”로 보수적으로 취급했다. 비공개 심사 전달이라는 사정만으로 재배포가 아니라고 추정하지 않는다. DACON에 보존기간, 접근자, 재이용·공개 여부를 별도로 확인해야 한다.

## 2. 공통 라이선스 해석

### ODC 계열

ODC-By 1.0은 데이터베이스의 구조·배열과 데이터베이스권을 다루며, 개별 Contents에는 별도 저작권·인격권·프라이버시권이 있을 수 있다고 명시한다. 데이터베이스 자체의 공유·변형·상업 이용은 가능하지만 공개 Convey 시 ODC-By, 라이선스/URL, notice를 제공해야 하고 Produced Work에는 데이터베이스 사용 attribution을 붙여야 한다. 위반 시 자동 종료와 제한된 복구 조항이 있다. [ODC-By 1.0 원문](https://opendatacommons.org/licenses/by/1-0/) (확인 2026-08-30).

VCC2020처럼 ODbL과 DbCL을 함께 제시하면 DbCL이 개별 contents에 대해 상업 이용과 sublicense를 포함한 저작권 허락을 준다. 반대로 ASVspoof 2019/2021 페이지가 ODC-By만 표시할 때는 개별 WAV/FLAC까지 자동 허용됐다고 보면 안 된다. [VCC2020 official repository](https://github.com/nii-yamagishilab/VCC2020-database), [ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/), [DbCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/) (확인 2026-08-30).

### Creative Commons

CC BY 4.0은 공유와 변형을 허용하지만 creator/attribution party, copyright notice, license notice/link, source URL과 변경 표시를 보존해야 한다. CC BY-SA 4.0은 공유한 adaptation에 동일한 SA 계열 또는 공식 compatible license를 요구한다. DACON 운영진은 CC BY-NC-SA 자료도 누구나 접근 가능한 공개 자원이고 최소 비영리 사용이 허용되며 라이선스 조건을 지키면 이 대회에서 사용할 수 있다고 공식 답변했다. 따라서 이 감사에서는 **CC BY-NC와 CC BY-NC-SA의 NC 조건 자체를 대회 사용 결격으로 보지 않는다.** 다만 대회 이후 상업 재사용은 별도이고, 귀속·NC·SA·변경 표시와 개별 사이트의 추가 이용조건은 계속 지켜야 한다. [DACON CC BY-NC-SA 공식 답변](https://dacon.io/competitions/official/236749/talkboard/417212), [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode), [CC BY-SA 4.0 legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode), [CC BY-NC 4.0 legal code](https://creativecommons.org/licenses/by-nc/4.0/legalcode), [CC BY-NC-SA 4.0 legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (확인 2026-08-30).

저작인격권·퍼블리시티·프라이버시와 제3자 저작권은 CC 허락 밖에 남을 수 있다. 데이터셋의 중앙 라이선스가 화자 동의, 원곡, 작곡, 가사, 샘플의 모든 권리를 자동 정리한다고 보지 않았다. [CC BY 4.0 deed의 기타 권리 고지](https://creativecommons.org/licenses/by/4.0/) 및 [CC FAQ](https://creativecommons.org/faq/) (확인 2026-08-30).

### ShareAlike 혼합 충돌

CC 공식 호환 라이선스 목록상 BY-SA 4.0 adaptation의 기여분은 BY-SA 4.0/후속·ported·지정 호환 라이선스로만, BY-NC-SA 4.0 adaptation은 BY-NC-SA 4.0/후속·ported·지정 호환 라이선스로만 배포할 수 있으며 현재 BY-NC-SA 호환 비-CC 라이선스는 없다. 따라서 **WaveFake(BY-SA)를 MedleyDB(BY-NC-SA) 또는 라이선스가 BY-NC로 확인될 수 있는 FakeMusicCaps와 하나의 합성 오디오 adaptation으로 섞는 것은 보수적으로 충돌**한다. [CC Compatible Licenses](https://creativecommons.org/compatible-licenses/) (확인 2026-08-30).

운영 원칙은 `license-isolated mixing strata`다. 예: BY-SA 계층에는 CC0/CC BY/BY-SA만 넣고 결과를 BY-SA로 관리한다. BY-NC-SA 계층은 별도 디렉터리·manifest·모델 실험으로 격리한다. 단순히 두 파일을 같은 학습 폴더에 두는 것은 혼합 adaptation이 아니지만, waveform을 실제 합성·overlay·concatenate한 하나의 파일로 만들 때는 사전 호환성 검토가 필요하다.

## 3. 후보별 감사

### 3.1 ASVspoof 2021 DF — CONDITIONAL

**자산 단위.** 공식 페이지는 LA/PA/DF 데이터베이스를 ODC Attribution License로 공개하고 DF evaluation set을 Zenodo로 연결한다. DF Zenodo는 약 34.5 GB의 FLAC evaluation corpus이지만 Zenodo 레코드의 자체 Rights 표시는 명확하지 않아 공식 ASVspoof 페이지를 기준으로 했다. [ASVspoof 2021 official](https://www.asvspoof.org/index2021.html), [ASVspoof 2021 DF Zenodo](https://zenodo.org/records/4835108) (확인 2026-08-30).

**권리·변형·제출.** ODC-By가 데이터베이스의 추출·재이용·변형을 허용해도 개별 녹음 contents 권리까지 일괄 부여하지 않는다. DF는 ASVspoof 2019, VCC2018/2020 등 여러 출처와 공격을 결합하므로 source별 contents 권리 chain을 유지해야 한다. VCC2020은 ODbL+DbCL이라 상대적으로 명확하지만, 전체 DF에 동일한 명확성이 있다고 추정할 수 없다. [ODC-By](https://opendatacommons.org/licenses/by/1-0/), [VCC2020 license](https://github.com/nii-yamagishilab/VCC2020-database) (확인 2026-08-30).

**접근조건.** Zenodo 공개 다운로드이지만, 실제 취득 시 버전·DOI·파일 해시·ASVspoof notice를 고정해야 한다. 비공식 Hugging Face 재패키징은 서로 다른 라이선스 라벨을 붙이기도 하므로 사용하지 않는다.

**판정.** 학습·segment·re-encode 자체는 ODC-By의 database permission과 각 upstream permission 범위에서 가능할 여지가 크지만, DACON에 **개별 FLAC 또는 변형 파일 전부** 전달할 권리는 문서만으로 닫히지 않는다. ASVspoof/EURECOM에 “DACON 비공개 심사위원에게 전체 DF 음원 및 파생 segment/mix를 전달할 수 있는지” 서면 확인 전 CONDITIONAL이다. 사용 시 ODC-By 원문/URL, database attribution, 변경 내역, upstream별 manifest를 함께 제출한다.

### 3.2 ASVspoof 2019 LA/PA 및 real PA 비교

2019 LA/PA DataShare 레코드는 ODC-By를 사용하며 LA와 PA가 VCTK에서 파생됐다고 설명한다. ODC-By의 contents gap 때문에 일반 LA/PA도 **CONDITIONAL**이다. [ASVspoof 2019 DataShare](https://datashare.ed.ac.uk/items/31074a11-b6f6-4e92-a4ad-07093f8c0c45), [2019 license text](https://datashare.ed.ac.uk/server/api/core/bitstreams/60c7de6d-37d3-45c1-b52e-9f18b0338e47/content), [VCTK DataShare](https://datashare.ed.ac.uk/items/04ff7c17-bfaf-432d-ae0f-79bdeee24bf8/full) (확인 2026-08-30).

별도의 **ASVspoof 2019 real PA** bespoke license는 내부 research로 사용 분야를 제한하고 transfer를 금지하며 기간 종료 후 36개월 내 삭제 의무를 둔다. DACON 전체파일 제출과 직접 충돌하므로 **NO-GO**다. [ASVspoof 2019 real PA license PDF](https://www.asvspoof.org/Database_ASVspoof2019_real_PA_License.pdf) (확인 2026-08-30).

### 3.3 ASVspoof 2015 — GO(귀속·manifest 조건)

DataShare full metadata는 전체 데이터셋의 `dc.rights`를 CC BY 4.0으로 명시하고, genuine 106명과 10종 spoofing 계열을 training/development/evaluation으로 배포한다. 현재 DataShare 레코드는 공개 다운로드 파일과 별도 license.txt를 제공한다. 구 ASVspoof 안내에는 이메일로 계정 발급을 받으라는 절차가 남아 있으므로 실제 취득 시 DataShare의 현재 공개 경로만 사용하고 우회하지 않는다. [ASVspoof 2015 DataShare full record](https://datashare.ed.ac.uk/items/a714c9b8-4acd-43a2-ac45-68eebcd070de/full), [ASVspoof 2015 page](https://www.asvspoof.org/index2015.html) (확인 2026-08-30).

CC BY 4.0은 비영리 학습, segmentation, re-encoding, augmentation, mixing과 DACON 파일 전달을 허용한다. dataset creators, dataset title, DOI/source URL, CC BY 4.0 URL, 변경 내용을 파일별 또는 합리적 묶음 단위로 보존한다. 화자 인격권·동의의 국제 범위는 무보증이므로 음성을 사람 사칭·재식별 등 대회 밖 용도로 쓰지 않는다. 이 조건에서 **GO**이며, 2021보다 생성 공격이 오래됐다는 것은 모델링 한계이지 라이선스 흠결은 아니다.

### 3.4 WaveFake — GO(귀속·BY-SA 격리 조건)

**데이터.** Zenodo dataset은 104,885개 fake audio를 CC BY-SA 4.0으로 배포하고 reference real audio는 포함하지 않는다. datasheet는 영어 원음 LJSpeech를 public domain, 일본어 원음 JSUT를 CC BY-SA 계열로 설명하고, JSUT 저자에게 연구 목적으로 fake sample 배포 허락을 받았다고 기록한다. [WaveFake Zenodo](https://zenodo.org/records/5642694), [WaveFake datasheet](https://zenodo.org/record/5642694/files/datasheet.pdf) (확인 2026-08-30).

**코드.** 공식 GitHub code는 MIT이며 code 복제·수정·배포 시 copyright/permission notice를 보존한다. MIT가 음원 ZIP에 적용되는 것은 아니다. [WaveFake repository](https://github.com/RUB-SysSec/WaveFake), [MIT license](https://github.com/RUB-SysSec/WaveFake/blob/main/LICENSE) (확인 2026-08-30).

**변형·제출.** CC BY-SA는 segmentation, re-encoding, augmentation, redistribution을 허용한다. 공유되는 adaptation이면 BY-SA 4.0 또는 공식 호환 라이선스로 제공하고 attribution·license link·source·변경 표시를 유지하며 추가 법적/기술적 제한을 걸지 않는다. 단순 format shifting은 반드시 adaptation이 된다고 단정할 수 없지만 변경은 기록한다. overlay/mix는 adaptation으로 보수 처리한다. [CC BY-SA legal code](https://creativecommons.org/licenses/by-sa/4.0/legalcode) (확인 2026-08-30).

**판정.** DACON이 최소 비영리 허용 공개 자원을 사용할 수 있다고 확인했고, Zenodo는 dataset을 CC BY-SA 4.0으로 공개하며 datasheet는 JSUT 권리자의 연구목적 fake sample 배포 허락을 기록한다. 따라서 **GO(준수조건부)**다. DACON 전달본에도 attribution, license link, source, 변경 표시와 BY-SA 조건을 붙이고, BY-NC/BY-NC-SA 음원과 단일 waveform으로 섞지 않는다.

### 3.5 LibriSpeech — CONDITIONAL(권리 범위 점검 후 GO)

OpenSLR 공식 페이지는 LibriSpeech corpus를 CC BY 4.0으로 배포하며 LibriVox 오디오북에서 파생됐다고 명시한다. 공개 다운로드이며 별도 등록 약관은 표시되지 않는다. [OpenSLR 12](https://www.openslr.org/12) (확인 2026-08-30).

CC BY 4.0은 비영리 학습, segment/re-encode/mix, 심사위원에게 실제 파일 전달을 허용한다. OpenSLR/저자/논문/URL/license/변경 attribution을 붙인다. 다만 LibriVox는 녹음을 미국 public domain으로 취급하면서 미국 밖에서는 해당 관할의 상태를 확인하라고 명시한다. 원저작 도서, 번역, 낭독 실연·녹음 권리의 한국 내 상태를 title별로 선별해야 한다. [LibriVox public domain policy](https://librivox.org/pages/public-domain/), [LibriVox about](https://librivox.org/pages/about-librivox/) (확인 2026-08-30).

따라서 전체를 무검토 사용하는 것은 CONDITIONAL이다. 대안은 저작권 만료가 한국에서도 명백한 제목만 allowlist하거나 OpenSLR/LibriVox에 한국 대회 재배포 범위를 확인하는 것이다. 화자 재식별·사칭과 같은 범위 밖 사용은 금지한다.

### 3.5.1 LJSpeech — GO(공개 도메인·짝맞춤 대조군)

공식 배포 페이지는 13,100개 short clip, 약 24시간 분량의 LJSpeech를 public domain으로 공개하며 text·audio·metadata에 제한이 없다고 명시한다. [LJSpeech official](https://keithito.com/LJ-Speech-Dataset/) (확인 2026-08-30).

WaveFake datasheet는 영어 fake sample의 원천 real data가 LJSpeech라고 설명한다. 따라서 WaveFake의 영어 subset을 쓸 때 LJSpeech 원본 clip을 real control로 함께 넣어야 `WaveFake ZIP 출처/화자/녹음환경` 자체를 REAL/FAKE 정답으로 외우는 shortcut을 줄일 수 있다. public-domain notice, canonical URL, archive hash와 수행한 segment/resample/re-encode 변경은 의무가 아니더라도 재현성을 위해 보존한다. 한 명의 화자만 포함하므로 ASVspoof 2015 또는 권리 검증된 다화자 real corpus를 함께 쓰고, LJSpeech 하나만 일반적인 real speech 대표로 취급하지 않는다.

### 3.6 MedleyDB — CONDITIONAL(사전 서면 동의 필수)

공식 download page는 MedleyDB를 CC BY-NC-SA 4.0의 non-commercial research용으로 제시하면서, full/part 재게시를 consent 없이 하지 말아 달라고 별도로 요청하고 Zenodo access request 경로를 둔다. 오디오는 participating artists의 원작에서 수정된 multitrack/stem이다. [MedleyDB downloads/terms](https://medleydb.weebly.com/downloads.html), [CC BY-NC-SA legal code](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) (확인 2026-08-30).

CC BY-NC-SA 자료의 대회 사용 자체는 DACON 공식 답변으로 허용 범위에 들어간다. 그러나 사이트의 no-republication request와 DACON 전체파일 전달이 겹친다. Rachel Bittner/NYU 측에 “학습 및 생성 fake 제작, DACON 심사위원에게 원본/파생 전체 파일 비공개 전달” 동의를 받기 전 **CONDITIONAL**이다.

동의 후에도 artist/track/title/license/source URL/changes를 트랙별로 유지하고 BY-NC-SA adaptation을 동일 계열로 공유한다. WaveFake BY-SA와 동일 waveform에 섞지 않는다. underlying composition, lyrics, performance, artist personality rights는 중앙 license 밖일 수 있으므로 특정 가수 모사 용도로 쓰지 않는다.

### 3.7 MUSDB18 — NO-GO(전체셋)

공식 페이지는 150 full-length tracks를 gated Zenodo로 제공하고 academic purposes only라고 설명한다. provenance는 DSD100/Mixing Secrets 100곡, MedleyDB 46곡, Native Instruments 2곡, Easton Ellises 2곡이다. 공식 tracklist에서 다수 DSD/Native Instruments 음원이 `Restricted`이고 일부만 CC BY-NC-SA 계열이다. [MUSDB18 official](https://sigsep.github.io/datasets/musdb.html), [official tracklist CSV](https://raw.githubusercontent.com/sigsep/website/master/content/datasets/assets/tracklist.csv) (확인 2026-08-30).

비영리·학술 목적 사용 자체는 대회 규정과 맞을 수 있지만, 다수 트랙의 `Restricted` 권리와 DACON 심사위원 파일 transfer 권한은 별개이며 parser/code의 라이선스도 음원 권리를 확장하지 않는다. 전체 MUSDB18은 **NO-GO**다. 개별 permissive track을 source 권리자에게 다시 확인해 쓰는 것은 이론상 가능하지만, MedleyDB 직접 허락 또는 FMA의 허용 라이선스 subset이 더 단순하다.

### 3.8 Meta AudioCraft / MusicGen — CONDITIONAL

**코드.** AudioCraft repository code는 MIT이므로 사용·수정은 가능하고 재배포 시 MIT notice를 보존한다. **가중치.** MusicGen model weights는 CC BY-NC 4.0이다. code MIT를 weights나 output에 적용해서는 안 된다. [AudioCraft](https://github.com/facebookresearch/audiocraft), [code LICENSE](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE), [weights LICENSE](https://github.com/facebookresearch/audiocraft/blob/main/LICENSE_weights) (확인 2026-08-30).

**출력물.** official model card는 research intended use, licensed sources로 학습했다는 설명, downstream 사용 전 risk assessment를 제시하지만 생성 WAV의 소유권·재배포 라이선스를 명시하지 않는다. “가중치가 CC BY-NC이므로 출력도 자동 CC BY-NC” 또는 그 반대를 추정하지 않는다. [MusicGen model card](https://github.com/facebookresearch/audiocraft/blob/main/model_cards/MUSICGEN_MODEL_CARD.md), [official Hugging Face model](https://huggingface.co/facebook/musicgen-small) (확인 2026-08-30).

**판정.** **MusicGen 가중치는 GO(준수조건부)**다. CC BY-NC 조건과 attribution/license/changes를 보존하면 대회에서 inference에 사용할 수 있다. 그러나 **생성 WAV는 여전히 CONDITIONAL**이다. 공식 문서가 output 소유권·재배포 조건을 명시하지 않으므로, Meta에 생성 WAV를 학습 데이터로 쓰고 DACON 심사위원에게 전달 가능한지와 필요한 notice를 확인해야 한다. melody conditioning은 입력 melody 권리를 별도로 확보하지 않으면 사용하지 않고 original text prompt만 사용한다. 모델 ID/revision, prompt, seed, inference params, output hash를 기록한다.

### 3.9 FakeMusicCaps v2 — NO-GO(명시 라이선스 전)

Zenodo v2는 27,605개의 10초 synthetic tracks와 약 12.9 GB audio를 설명하지만 record의 Rights/License 필드가 비어 있다. GitHub repository에는 CC BY-NC 4.0 LICENSE가 있으나 별도 Zenodo 데이터 ZIP까지 그 범위를 확장한다고 명시하지 않는다. [FakeMusicCaps Zenodo](https://zenodo.org/records/15063698), [FakeMusicCaps GitHub](https://github.com/polimi-ispl/FakeMusicCaps), [GitHub LICENSE](https://github.com/polimi-ispl/FakeMusicCaps/blob/main/LICENSE) (확인 2026-08-30).

논문은 MusicCaps captions를 여러 text-to-music model(MusicGen, MusicLDM, AudioLDM2, Stable Audio Open, Mustango)에 입력했다고 설명한다. 각 generator의 output terms와 captions의 제3자 권리 chain이 서로 다르다. [FakeMusicCaps paper/PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12295441/) (확인 2026-08-30).

비공식 Hugging Face mirror가 Apache-2.0 metadata를 표시하더라도 원 음원을 재라이선스할 권한의 증거가 아니며, 해당 mirror는 parquet path/metadata 중심이다. [DeepFense mirror](https://huggingface.co/datasets/DeepFense/FakeMusicCaps) (확인 2026-08-30). Zenodo audio ZIP에 적용되는 명시 라이선스, 모델별 output permission, MusicCaps caption permission, DACON judge transfer를 authors가 서면 확인하기 전 **NO-GO**다. 확인 후에도 BY-NC라면 WaveFake BY-SA와 단일 mix를 만들지 않는다.

## 4. 더 안전한 음악 대안

### Stable Audio Open Small — CONDITIONAL이나 MusicGen보다 출력권이 명확

공식 Hugging Face model은 gated access로 contact information 공유와 license/privacy 동의를 요구하며, 0.5B parameter, 최대 11초 stereo 출력이다. Community License는 research/non-commercial 사용과 연매출 100만 달러 미만의 제한적 commercial use를 다루고, commercial use면 Stability AI registration을 요구한다. 당사자 사이에서는 생성 output을 법이 허용하는 범위에서 사용자가 소유한다고 명시한다. [Stable Audio Open Small model card](https://huggingface.co/stabilityai/stable-audio-open-small), [Community License](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE) (확인 2026-08-30).

단, model/material/output 사용은 법령·문서·Stability AUP를 따라야 하고 foundation generative AI model을 생성·개선하는 데 쓰는 것이 제한된다. 위반 종료 시 Stability materials와 derivative works를 삭제해야 하며 IP 소송 제기 시 종료 조항도 있다. 생성 output 자체는 model의 Derivative Work 정의에서 제외되지만 제3자 권리 비침해를 보증하지 않는다. [Community License](https://huggingface.co/stabilityai/stable-audio-open-small/blob/main/LICENSE), [Stability AUP](https://stability.ai/use-policy) (확인 2026-08-30).

대회 detector 학습이 “foundational generative AI model 개선”은 아니라고 보이지만 이는 추론이므로 DACON output 전달과 해당 제한의 적용 여부는 확인한다. 비영리 사용이 가능하다는 전제에서는 상금의 commercial 분류는 별도 장애로 잡지 않는다. 실제 참가 주체가 gated 약관에 동의하고 license/AUP snapshot을 보존한 뒤 사용하는 **CONDITIONAL(접근조건 충족 후 사용 가능)**이며, output 조항이 없는 MusicGen보다 권리 구조는 낫다.

### FMA permissive subset — CONDITIONAL(트랙별 검증 후 GO 가능)

FMA repository는 code를 MIT, metadata를 CC BY 4.0으로 제공하지만 오디오 저작권은 보유하지 않으며 **각 artist가 고른 트랙별 license**로 배포한다고 명시한다. `tracks.csv`에 track metadata가 있으므로 전체 ZIP을 일괄 허용된 데이터로 취급하면 안 된다. [FMA official repository](https://github.com/mdeff/fma), [FMA code license](https://github.com/mdeff/fma/blob/master/LICENSE.txt), [FMA paper](https://arxiv.org/abs/1612.01840) (확인 2026-08-30).

실무 allowlist는 현재 source URL이 살아 있고 license가 CC0, CC BY, CC BY-SA, CC BY-NC, CC BY-NC-SA인 트랙을 취할 수 있다. ND, custom/restricted/unknown은 제외한다. WaveFake와 mix할 BY-SA 계층에는 CC0/BY/BY-SA만 넣고, NC/NC-SA 트랙은 별도 계층에 둔다. artist/title/track URL/license URL/download date/hash/변경을 파일별 manifest에 고정하고 SA adaptation은 해당 계열 조건으로 관리한다. FMA archive의 오래된 track metadata 변동 가능성 때문에 다운로드 전 source별 확인이 필요하여 현 단계는 **CONDITIONAL**이다.

## 5. 권장 최소 안전 조합

### 즉시 준비 가능한 보수 조합

1. fake speech: **ASVspoof 2015** — GO, CC BY attribution/changes/manifest.
2. matched real speech: **LJSpeech** — GO, WaveFake 영어 subset의 paired real control로 사용.
3. real speech diversity: **LibriSpeech allowlist** — 한국에서도 원저작 만료가 명백한 title만; 확인 전 CONDITIONAL.
4. 추가 fake speech: **WaveFake** — GO, attribution·BY-SA·변경 표시를 적용하고 BY-SA 계층에서 사용.
5. real music: **FMA verified subset** — CC0/BY/BY-SA 및 NC/NC-SA 트랙까지 사용할 수 있으나 트랙별 현행 라이선스를 검증하고 호환 계층별로 분리.
6. fake music: **Stable Audio Open Small** — 참가 주체의 gated consent, AUP, output judge transfer 확인 후 생성.

이 구조는 대규모 모델을 직접 학습하지 않으면서도 출력권이 불명확한 MusicGen보다 문서화가 쉽다. 다만 음악 두 축은 아직 서면·트랙 검증이 끝나지 않아 “완전 GO”가 아니다.

### 제외·보류

- 즉시 제외: **MUSDB18 전체**, **ASVspoof 2019 real PA**, **FakeMusicCaps v2**.
- 권리자 확인 전 보류: **ASVspoof 2021 DF/2019 LA·PA**, **MedleyDB**, **MusicGen 생성 WAV**, **Stable Audio Open Small**.
- ASVspoof 2021을 꼭 쓸 경우: DF 전체 contents 전달 허락과 upstream provenance manifest 확보 후 2015와 별도 stratum으로 추가.
- MedleyDB를 꼭 쓸 경우: no-republication 요청과 judge 전달 consent를 확인한 뒤 BY-NC-SA 전용 stratum에서만 사용.

## 6. 데이터 취득 전에 반드시 통과할 gate

각 asset은 다음 필드가 전부 채워지기 전 다운로드·학습에 넣지 않는다.

- canonical official URL, version/revision/DOI, acquisition date, archive filename/hash
- asset unit별 license: dataset audio, metadata, code, model weights, API, generated output
- access/registration account와 동의 주체, 당시 terms snapshot/PDF/hash
- source creator/title/license URL, 변경 내역, attribution text
- upstream/third-party source와 speaker/artist/work/title
- allowed purpose: 최소 비영리 사용 허용 여부(DACON 대회 적격성 확인됨)
- allowed acts: train, segment, re-encode, augment, mix, generate, judge transfer
- DACON package에 실제 파일을 넣을 수 있다는 근거 또는 권리자 이메일
- NC/SA/notice/source/offering obligations와 termination/deletion trigger
- license stratum: `permissive-by`, `by-sa`, `by-nc-sa`, `restricted-do-not-use`

파일별 `source_id`를 training manifest, preprocessing log, final submission bundle까지 유지한다. 생성물은 prompt/seed/model revision/parameters/output hash를 함께 보존한다. 원본과 adaptation을 구분하고, 학습에 실제 사용하지 않은 파일은 제출 패키지에서도 분리한다.

## 7. 조사 방법과 한계

공식 대회 페이지, 권리자/배포기관 페이지, 공개 라이선스 원문, 공식 repository/model card를 우선했다. Bright Data live-research workflow도 시도했으나 이 환경의 네트워크 호출이 실패해, 공식 웹 검색과 직접 페이지 열람으로 교차 확인했다. 이 보고서는 파일 안의 개별 license/consent 문서를 실제 다운로드 후 hash 검증하지 않았으므로 **취득 시점 재검증이 필수**다. 기준일 이후 terms·데이터 버전·DACON FAQ가 변경될 수 있다.
