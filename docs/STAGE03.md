# Stage 03 — 온라인 탐지와 추론 구현 비교

## 현재 실행 범위

두 백본의 고정 특징 추출을 온라인 `[t−15,…,t]` 창으로 시작했다. R01–R04의 정상 fit/calibration과 테스트를 순서대로 처리한다.
정상 fit 초기 구간만 첫 프레임 왼쪽 패딩을 허용하며, 검증·테스트는 실제 과거 16프레임이 확보된 뒤 시작한다. 정확도 비교의 공통 대상은 오프라인과 같은 `t=19..N−8`이다.

R01–R02 두 백본과 R03 V-JEPA의 온라인 학습·평가를 3 seeds 모두 완료했다. 다른 조건은 진행 중이며, 실제 FPS·알람 지연은 아직 측정하지 않았다.

## R01 오프라인·온라인 정확도

두 모드 모두 테스트 15개 영상의 같은 3,295프레임(이상 1,227프레임)에서 평가했다. 아래는 seed 0/1/2 지표의 평균이다.

| 백본 | 구성 | 오프라인 AUROC / AP (%) | 온라인 AUROC / AP (%) |
|---|---|---:|---:|
| DINOv3-L | P0 전역 메모리 | 78.51 / 57.30 | 79.00 / 58.02 |
| DINOv3-L | P1 위상 hard | 45.21 / 32.32 | 54.79 / 36.94 |
| DINOv3-L | P2 위상 soft | 45.47 / 32.18 | 54.72 / 36.82 |
| DINOv3-L | P3 위상 soft + 시간 | 47.01 / 33.21 | 53.69 / 36.92 |
| V-JEPA 2.1-L | P0 전역 메모리 | 32.90 / 28.38 | 30.37 / 29.53 |
| V-JEPA 2.1-L | P1 위상 hard | 24.97 / 25.30 | 33.83 / 28.42 |
| V-JEPA 2.1-L | P2 위상 soft | 30.43 / 26.86 | 32.44 / 27.84 |
| V-JEPA 2.1-L | P3 위상 soft + 시간 | 37.58 / 28.83 | 34.60 / 29.72 |

V-JEPA P3 온라인−오프라인 차이는 AUROC −2.97pp(95% 영상 bootstrap CI −10.92..5.40pp), AP +0.89pp(−2.19..6.43pp)다. DINOv3는 AUROC +6.68pp(−0.29..14.00pp), AP +3.71pp(−0.57..8.85pp)다. 두 백본의 모드별 P3 차이는 모두 신뢰구간에 0을 포함하며 DINOv3 P3는 P0보다 낮다.

온라인 P3의 V-JEPA−DINOv3 AUROC 차이는 −19.09pp(−32.15..−7.87pp)다. R01에서는 DINOv3가 더 높았으나 이 한 장비 결과로 전체 장비·LoRA·실시간 조건의 우열을 판단하지 않는다. 원본 점수 CSV·정상 보정·실행별 지표는 [DINOv3 온라인](../results/stage02/dinov3-l/online/R01), [V-JEPA 온라인](../results/stage02/vjepa21-l/online/R01)에 공개한다.

![R01 온라인·오프라인 비교](figures/stage02/R01_mode_comparison.png)
![R01 온라인 구성요소 비교](figures/stage02/R01_online_variants.png)
![R01 온라인 seed 0 ROC와 PR](figures/stage02/R01_online_roc_pr.png)

<details>
<summary>R01 온라인 정상 위상 정렬과 고정 테스트 영상 03</summary>

![DINOv3 온라인 위상 정렬](figures/stage02/dinov3-l_R01_online_phase_alignment.png)
![V-JEPA 온라인 위상 정렬](figures/stage02/vjepa21-l_R01_online_phase_alignment.png)
![DINOv3 온라인 시계열](figures/stage02/dinov3-l_R01_online_sequence03.png)
![V-JEPA 온라인 시계열](figures/stage02/vjepa21-l_R01_online_sequence03.png)

</details>

![R01 V-JEPA 온라인 위상 학습](figures/stage02/vjepa21-l_R01_online_seed0_phase.png)

## R02 오프라인·온라인 정확도

테스트 15개 영상에서 동일한 **9,210프레임(이상 2,949)**을 비교했다. R02/12·13·14의 라벨 정렬 미확정 18프레임은 양쪽 모드에서 같은 마스크로 제외했다. 정확한 정렬이 확인됐다는 의미는 아니며, ±1 인덱스 가정과 오프셋별 민감도는 [Stage 00](STAGE00.md)에 기록한다.

| 백본 | 구성 | 오프라인 AUROC / AP (%) | 온라인 AUROC / AP (%) |
|---|---|---:|---:|
| DINOv3-L | P0 전역 메모리 | 73.32 / 62.09 | 71.99 / 60.53 |
| DINOv3-L | P1 위상 hard | 72.20 / 59.04 | 72.44 / 58.70 |
| DINOv3-L | P2 위상 soft | 71.98 / 60.68 | 72.72 / 60.47 |
| DINOv3-L | P3 위상 soft + 시간 | 79.84 / 65.11 | 79.89 / 63.73 |
| V-JEPA 2.1-L | P0 전역 메모리 | 58.19 / 36.98 | 65.17 / 42.82 |
| V-JEPA 2.1-L | P1 위상 hard | 57.18 / 36.65 | 65.62 / 41.90 |
| V-JEPA 2.1-L | P2 위상 soft | 59.71 / 38.79 | 67.05 / 44.91 |
| V-JEPA 2.1-L | P3 위상 soft + 시간 | 63.10 / 42.55 | 67.17 / 45.64 |

V-JEPA P3의 온라인−오프라인 차이는 AUROC **+4.07pp(95% paired 영상 bootstrap CI +0.26..+8.45pp)**, AP **+3.09pp(+0.14..+5.56pp)**다. DINOv3는 AUROC +0.06pp(−3.57..+4.69pp), AP −1.37pp(−5.34..+3.02pp)로 모드 차이의 CI에 0이 포함된다. R02에서의 V-JEPA 모드 차이를 전체 장비의 온라인 우위 또는 실시간 성능 향상으로 해석하지 않는다.

온라인 P3의 V-JEPA−DINOv3 차이는 AUROC −12.72pp(−23.97..−0.58pp), AP −18.09pp(−36.36..+0.13pp)다. 원본 [DINOv3 R02 온라인](../results/stage02/dinov3-l/online/R02), [V-JEPA R02 온라인](../results/stage02/vjepa21-l/online/R02)과 [전체 paired CI](../results/stage02/device_summary.json)를 제공한다.

![R02 온라인·오프라인 정확도 비교](figures/stage02/R02_mode_comparison.png)
![R02 온라인 구성요소 비교](figures/stage02/R02_online_variants.png)
![R02 온라인 seed 0 ROC와 PR](figures/stage02/R02_online_roc_pr.png)

<details>
<summary>R02 온라인 정상 위상 정렬과 고정 테스트 영상 03</summary>

![DINOv3 R02 온라인 위상 정렬](figures/stage02/dinov3-l_R02_online_phase_alignment.png)
![V-JEPA R02 온라인 위상 정렬](figures/stage02/vjepa21-l_R02_online_phase_alignment.png)
![DINOv3 R02 온라인 시계열](figures/stage02/dinov3-l_R02_online_sequence03.png)
![V-JEPA R02 온라인 시계열](figures/stage02/vjepa21-l_R02_online_sequence03.png)

</details>

## R03 V-JEPA 오프라인·온라인 정확도

테스트 17개 영상의 동일한 **11,563프레임(이상 4,922)**에서 V-JEPA의 seed 0/1/2를 비교했다. 온라인 정상 fit은 2,618 clip, 정상 검증은 2,771 clip이며, 세 실행 모두 최소 정상 검증 CE에 따라 epoch 20을 선택했다. 선택 모델의 정상 원형 MAE는 5.35/5.67/6.09% 주기다.

| 구성 | 오프라인 AUROC / AP (%) | 온라인 AUROC / AP (%) |
|---|---:|---:|
| P0 전역 메모리 | 45.67 / 37.98 | 46.40 / 38.36 |
| P1 위상 hard | 50.38 / 41.57 | 48.90 / 41.51 |
| P2 위상 soft | 50.49 / 42.74 | 50.84 / 43.46 |
| P3 위상 soft + 시간 | 50.45 / 43.53 | 50.80 / 44.29 |

온라인 P3의 AUROC는 **50.80%(95% 영상 bootstrap CI 37.67..62.00)**, AP는 **44.29%(28.94..63.39)**다. P3−P0 차이는 AUROC +4.40pp(−1.33..+11.24pp), AP +5.93pp(+1.08..+12.48pp)다. AP 차이가 양수여도 절대 AP는 이상 프레임 비율 42.57%와 가깝고 AUROC는 50% 부근이므로, 탐지 성능이 제한적이다.

P3의 온라인−오프라인 차이는 AUROC +0.35pp(−2.61..+3.87pp), AP +0.75pp(−1.18..+3.80pp)로 두 CI에 0이 포함된다. R03에서는 온라인 모드의 정확도 개선을 확인하지 못했다. DINOv3의 R03 온라인 평가와 전체 온라인 Macro4는 이 공개 집계에 포함되지 않았다. 원본 [V-JEPA R03 온라인](../results/stage02/vjepa21-l/online/R03)과 [paired CI](../results/stage02/device_summary.json)를 제공한다.

![R03 V-JEPA 온라인·오프라인 비교](figures/stage02/R03_mode_comparison.png)
![R03 V-JEPA 온라인 구성요소 비교](figures/stage02/R03_online_variants.png)
![R03 V-JEPA 온라인 seed 0 ROC와 PR](figures/stage02/R03_online_roc_pr.png)

<details>
<summary>R03 V-JEPA 정상 위상 학습·정렬과 고정 테스트 영상 03</summary>

![R03 V-JEPA 온라인 정상 위상 학습](figures/stage02/vjepa21-l_R03_online_seed0_phase.png)
![R03 V-JEPA 온라인 정상 위상 정렬](figures/stage02/vjepa21-l_R03_online_phase_alignment.png)
![R03 V-JEPA 온라인 시계열](figures/stage02/vjepa21-l_R03_online_sequence03.png)

</details>

R03 DINOv3와 R04 V-JEPA의 온라인 위상 head 학습도 각각 3 seeds·20 epoch를 완료했다. [정상 위상 로그](../results/stage02/phase_summary.csv)의 45개 실행은 정상 데이터 학습의 범위이며, 미완료 테스트 평가를 포함한 이상탐지 성능 집계가 아니다.

<details>
<summary>추가 완료된 정상-only 위상 head 학습</summary>

![R03 DINOv3 온라인 정상 위상 학습](figures/stage02/dinov3-l_R03_online_seed0_phase.png)
![R04 V-JEPA 온라인 정상 위상 학습](figures/stage02/vjepa21-l_R04_online_seed0_phase.png)

</details>

## DINO 과거 프레임 특징 재사용 검증

DINO는 각 프레임을 독립적으로 인코딩하므로, 이전 프레임의 특징을 유지하고 도착한 프레임만 추가하는 구현을 별도로 작성했다. 16개 프레임 특징을 넘겨 보관하지 않고, 위상 입력의 전체 창 평균과 마지막 두 프레임 패치 평균을 유지한다.
CPU 테스트에서 순차 입력·warmup·초기 패딩·창 평균·한 번만 인코딩하는 동작을 검증했다.

실제 R01 정상 영상의 네 온라인 clip에서 전체 클립 추론과 비교했다. [원본 수치](../results/setup/dinov3_streaming.json)는 성공 여부와 오차를 각각 기록한다.

| 정밀도 | 패치 상대 L2 오차 | 전역 상대 L2 오차 | 검증 |
|---|---:|---:|---|
| BF16 | 0.757–0.890% | 0.401–0.490% | 사전 수치 기준 0.1% 초과 |
| FP32 | 약 0.00018% | 약 0.00005–0.00008% | 사전 수치 기준 0.001% 이내 |

FP32에서는 평균 동작이 일치했다. BF16에서 배치 구성에 따른 수치 차이가 있으므로, **주 정확도 실험은 두 모드 모두 기존 전체 클립 BF16 추론을 사용한다**. 재사용 구현으로 생성하는 캐시는 위 수치 검증이 통과되지 않으면 생성 도구가 거부한다.
현재의 특징 비교를 점수·알람의 동등성이나 실시간 속도 검증으로 해석하지 않는다. 최적화 런타임 결과를 공개하려면 실제 점수·알람 차이도 확인하고 사용 정밀도를 명시해야 한다.

## 후속 런타임 측정 범위

30 FPS 입력 도착, FIFO/no drops, warmup 19, 3프레임 연속 초과 알람, 고정 테스트 메모리 조건을 사용한다.
입력 decode/resize, encoder, 위상 head, PCA·메모리 검색, 점수·알람, 대기열, 오프라인 7프레임 lookahead 대기를 분리해 측정한다.
전체 지연 p50/p95, 지속 FPS, peak VRAM, 대기열 증가, 이벤트 탐지 지연·누락·오탐을 그래프로 제공한다. 캐시 생성 시간과 GPU cold forward 시간은 이 측정에 해당하지 않는다.

```bash
python -m ipad_jepa.features --model dinov3-l --mode online \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --upstream third_party/dinov3 --data-root "$IPAD_DATA_ROOT" \
  --devices R01 R02 R03 R04 --splits fit calibration test --batch-size 4 --workers 2

python scripts/verify_dino_streaming.py \
  --sequence "$IPAD_DATA_ROOT/R01/training/frames/01" \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

두 번째 명령은 현재 BF16 수치 기준 실패를 JSON에 기록한 뒤 비정상 종료한다. 원인을 숨기거나 성공 기준을 바꿔 통과 처리하지 않는다. 전체 행렬과 완료 조건은 [experiment_matrix.yaml](../configs/experiment_matrix.yaml)에 있다.
