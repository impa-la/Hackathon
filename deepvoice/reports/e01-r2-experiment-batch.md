EXPERIMENT_BATCH: BLOCKED_RESOURCE

# E01-R2 log-mel CNN reference baseline 실행 보고

## 판정

코드·실데이터 loader·GPU smoke는 통과했지만, 고정 workload의 실측 runtime projection이 3 GPU-hour gate를 넘어서 전체 3-seed 학습은 시작하지 않았다. 축소 seed·축소 split 결과를 E01 성능으로 주장하지 않는다.

## 고정 계약

- manifest SHA-256: `2f900e53cb728571f330ae24f885d6e6fade8c3ba61b5388fd9b6a4b28792ec6`
- manifest integrity row count: 137,328
- split crossing group: 0
- test isolation: crossing 검사용 `content_group_key`, `recommended_content_split`만 투영; test 통계 0건
- segment: 8초, 파일당 최대 8개, 짧은 파일은 explicit valid-sample/frame mask를 정규화와 CNN pooling에 적용
- feature: waveform만 사용; source/codec/rate/channel/path/provider metadata는 입력 금지
- sampling: group-first speech/music 50:50, real/fake 균형, paired speech content, AIME provider 균형
- seeds: 20260830 / 20260831 / 20260832

## 환경 및 검증

- preflight: READY (0 blockers)
- actual locator decode: 4/4 PASS
- unit/contract tests: 9/9 PASS
- GPU tiny smoke: PASS, 6.797 segment/s, peak 64.65 MiB
- GPU autotune: batch 16, 440.658 segment/s, peak 0.120 GiB
- balanced real-locator pilot: 64 samples, 17.321 sample/s, peak 0.119 GiB
- balanced loader probe: 21.587 file/s

## 자원 판정

- workload: 32,768 samples/epoch × 20 epochs × 3 seeds = 1,966,080 training decodes
- projected three-seed wall/GPU time: 32.088 hours
- gate: 3.0 GPU-hours and 24.0 wall-hours
- runtime gate status: BLOCKED_RESOURCE
- monetary cost: local existing hardware, incremental API/cloud cost KRW 0

## 재현성

- config SHA-256: `83fd9c9660c13856351332fb63c68f6eadcf77f350b39529fca933461ea6d1a9`
- E01 code inventory SHA-256: `3f2bd7e032660273dd430263db9d27e54fbe5011bbb57b26dea9b7545dcfc97e`
- E00-R2 contract SHA-256: `b489b136eb80edba8e8a5d6636ae70e273bb677201a6e7da0b3dfb48a2aae4ce`
- git HEAD: `4eff360b862d755fc4b06582f93740ec4bb1bda4`
- run UTC: `2026-08-30T11:47:38.846501+00:00`

## 산출물 범위

- 실제 validation prediction/OOF/metric/checkpoint는 생성하지 않았다.
- tiny smoke와 pilot loss는 합성·소규모 실행 건전성 검사이며 E01 성능 결과가 아니다.
- E02 학습은 수행하지 않았다.
- E01 R1 blocked evidence는 보존하며, R1 pilot의 nonfinite loss를 이 revision이 supersede한다.

## 환경 재구성 명령

```powershell
py -3.13 -m venv deepvoice\.venv-e01
deepvoice\.venv-e01\Scripts\python.exe -m pip install --upgrade pip
deepvoice\.venv-e01\Scripts\python.exe -m pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
deepvoice\.venv-e01\Scripts\python.exe -m pip install pyarrow==25.0.1
```

공식 설치 근거: https://pytorch.org/get-started/previous-versions/ , https://arrow.apache.org/docs/python/install.html
