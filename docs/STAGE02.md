# Stage 02 — 고정 백본 특징 추출과 정상 위상 학습

## 완료 범위

DINOv3-L와 V-JEPA 2.1-L의 R01 오프라인 정상 학습 23개 영상과 정상 검증 5개 영상에서 특징을 추출했다.
학습은 4프레임 간격의 1,248개 clip, 검증은 1프레임 간격의 1,087개 clip이다. 테스트 영상과 진단 영상은 사용하지 않았다.
각 백본의 seed 0/1/2에서 20 epoch 위상 예측기 학습을 완료했다. R01의 PCA·메모리 구성과 이상탐지 평가도 완료했으며 아래에 기록했다. R02–R04 및 온라인 비교는 아직 수행 중 또는 전이다. Stage 02 완료 태그는 생성하지 않는다.

## 정상 검증 결과와 한계

| 백본 | seed | 선택 epoch | 정상 검증 CE | 선택 모델 원형 MAE (%주기) | 마지막 epoch 원형 MAE (%주기) |
|---|---:|---:|---:|---:|---:|
| DINOv3-L | 0 | 1 | 5.2971 | 24.21 | 12.32 |
| DINOv3-L | 1 | 1 | 5.2968 | 22.34 | 12.83 |
| DINOv3-L | 2 | 1 | 5.2989 | 23.96 | 11.41 |
| V-JEPA 2.1-L | 0 | 1 | 5.3442 | 23.52 | 15.43 |
| V-JEPA 2.1-L | 1 | 1 | 5.3432 | 23.88 | 15.09 |
| V-JEPA 2.1-L | 2 | 1 | 5.3488 | 22.93 | 13.70 |

CE는 200개 상대 위상 클래스에 대한 교차 엔트로피다. 균일 확률의 CE는 `ln(200)=5.2983`이며, 선택된 모델들의 CE가 이 기준과 비슷하거나 더 높다.
원형 MAE는 `mean(abs(wrap(predicted_phase − t/N)))`로 계산하며 %주기는 그 값의 100배다. 이는 공정 단계의 수작업 의미 라벨과 비교한 값이 아니다.

학습 후반에 원형 MAE는 감소하지만 CE 기준에서는 모든 실행의 첫 epoch가 선택됐다. 선택된 모델의 위상 예측은 약하며, 후반 MAE만으로 모델 품질을 평가하면 선택된 모델의 결과를 과대평가하게 된다.
사전 계획대로 정상 검증 CE에 따라 선택했다. 이 결과만으로 두 백본의 이상탐지 정확도 또는 실시간성을 판단할 수 없다. 추가 장비와 정상 위상 학습 진단이 필요하다.

![DINOv3 정상 위상 학습](figures/stage02/dinov3-l_R01_offline_seed0_phase.png)
![V-JEPA 정상 위상 학습](figures/stage02/vjepa21-l_R01_offline_seed0_phase.png)

seed 0 곡선은 대표 실행의 전체 epoch를 보여준다. 나머지 seed의 수치도 [phase_summary.csv](../results/stage02/phase_summary.csv)와 실행별 CSV/JSON에 공개한다. 처리량과 실제 알람 지연 결과는 아직 없다. R01 이상탐지 지표는 아래 절에 기록했다.

## 캐시와 입력 검증

- RGB 전체 프레임을 PIL bilinear로 384×384 resize, ImageNet 평균/표준편차 정규화.
- 오프라인 입력은 `[t−8,…,t+7]`, 정상 학습의 위상 감독은 `floor(200*t/N)`.
- 패치 특징은 `576×1024`, 위상 입력은 `1024`. 백본 고정, BF16 추론 후 캐시는 FP16 저장.
- 모델 SHA256, upstream commit, 어댑터/읽기 코드 해시, 전처리, 모드, 원본 JPG 콘텐츠/파일명 해시, 분할과 target 인덱스를 캐시 메타데이터에 기록.
- 학습 로더는 정상 training 영상의 fit/calibration 분할만 허용하며, 변경된 원본·혼합 가중치·분할 불일치를 거부한다.
- 온라인 학습에 한해 첫 프레임 왼쪽 패딩을 허용한다. 온라인 검증/테스트 입력은 16개 실제 과거 프레임이 확보된 후 시작한다.

## 재현

```bash
python -m ipad_jepa.features --model dinov3-l --mode offline \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --upstream third_party/dinov3 --data-root "$IPAD_DATA_ROOT" \
  --devices R01 --splits fit calibration --batch-size 4 --workers 2

python -m ipad_jepa.train_phase --cache artifacts/features/dinov3-l/offline \
  --device R01 --seed 0 --out artifacts/runs/dinov3-l/offline/R01/seed0

python scripts/export_phase_results.py
python scripts/plot_phase_training.py \
  --results results/stage02/dinov3-l/offline/R01/seed0 \
  --out docs/figures/stage02/dinov3-l_R01_offline_seed0_phase
```

V-JEPA는 `--model vjepa21-l`, `--upstream third_party/vjepa2`, `--weights artifacts/weights/vjepa2_1_vitl_dist_vitG_384.pt`를 사용한다. seed 1/2는 별도 출력 경로로 실행한다.
위상 예측기는 1024→256→200, ReLU/dropout 0.1, AdamW 학습률 1e-3/weight decay 1e-4, batch 256, 20 epoch다. 정상 검증 CE 최저 checkpoint만 선택한다.
모델·특징 캐시·학습 checkpoint는 로컬 `artifacts`에 보관한다. 공개 파일은 코드·CSV·JSON·PNG·SVG뿐이다.

## R01 실제 이상탐지 평가: 고정 백본, 오프라인, 3 seeds

R01 테스트 15개 영상의 동일한 3,295프레임(이상 1,227프레임)에서 평가했다. 공통 대상은 `t=19..N−8`이다.
PCA·메모리는 정상 fit 23개 영상, 온도·median/MAD·임계값은 정상 검증 5개 영상으로만 구성했다. 테스트 라벨은 점수 계산이 끝난 후 지표 계산에 사용했다.

| 구성 | DINOv3 AUROC (%) | DINOv3 AP (%) | V-JEPA AUROC (%) | V-JEPA AP (%) |
|---|---:|---:|---:|---:|
| P0 전역 메모리 / hard | 78.51 | 57.30 | 32.90 | 28.38 |
| P1 위상 메모리 / hard | 45.21 | 32.32 | 24.97 | 25.30 |
| P2 위상 메모리 / soft k=5 | 45.47 | 32.18 | 30.43 | 26.86 |
| P3 P2 + 시간 점수 | 47.01 | 33.21 | 37.58 | 28.83 |

표는 seed 0/1/2의 지표 평균이며, 점수를 seed 간 평균해서 계산한 값이 아니다. 아래 오차막대는 동일 영상 단위 1,000회 재표집의 95% percentile 신뢰구간이다.
R01에서는 DINOv3 P0가 가장 높았고, 위상 조건을 추가하면 하락했다. DINOv3 P3−P0 AUROC 차이는 −31.49pp(95% CI −42.90..−19.11pp)다.
V-JEPA P3−P0 차이는 +4.67pp(−1.57..12.00pp)로, 이 장비의 표본에서는 개선을 확정할 수 없다. P3의 V-JEPA−DINOv3 차이는 −9.44pp(−18.21..−0.64pp)다.
한 장비의 결과이므로 네 장비 Macro AUROC, 일반적인 백본 우열, 온라인 또는 실시간 성능으로 확대 해석하지 않는다.

![R01 구성요소 비교](figures/stage02/R01_offline_variants.png)
![R01 seed 0 ROC와 PR](figures/stage02/R01_offline_roc_pr.png)

현재 설정에서 선택된 위상 예측기가 약하며, 정상 검증의 상대 위치와 예측 위상이 잘 맞지 않는다. 위상 조건으로 잘못된 정상 메모리 후보를 제한했을 가능성이 있다.
V-JEPA의 P0도 낮으므로 위상 예측만으로 성능 하락 전체를 설명할 수는 없다. 실제 이상 유형별 효과는 후속 진단과 나머지 장비에서 확인해야 한다.
이상 점수는 큰 값이 이상이라는 사전 정의를 유지했으며, 테스트 결과를 보고 점수 부호를 뒤집거나 epoch·온도·임계값을 다시 선택하지 않았다.

<details>
<summary>정상 위상 정렬과 고정 테스트 영상 03의 점수·알람</summary>

![DINOv3 정상 위상 정렬](figures/stage02/dinov3-l_R01_offline_phase_alignment.png)
![V-JEPA 정상 위상 정렬](figures/stage02/vjepa21-l_R01_offline_phase_alignment.png)
![DINOv3 점수 시계열](figures/stage02/dinov3-l_R01_offline_sequence03.png)
![V-JEPA 점수 시계열](figures/stage02/vjepa21-l_R01_offline_sequence03.png)

</details>

시계열의 가로축은 target 프레임이며 실제 벽시계 탐지 시간은 아니다. 3프레임 연속 초과 알람을 표시하지만 대기열·전처리·오프라인 lookahead 대기를 측정한 실시간 알람 지연 결과는 아직 없다.

## 메모리·보정·지표 재현

각 위상 bin의 후보는 영상별 같은 수를 목표로 표본 추출하고 부족한 영상의 할당량만 재분배한다. bin당 최대 10,000패치, 총 최대 160,000 후보를 사용하며 PCA 50,000 표본도 영상×위상 strata별로 균등 선택한다.
PCA 256차원/비 whitening/L2 정규화, bin 16개×128 prototype=2,048개, greedy k-center다. P0도 동일한 prototype 2,048개를 사용하고 bin tag만 무시한다.
정상 검증에서 영상별 균등 표본 최대 50,000패치의 5번째 이웃 제곱 거리 중앙값으로 soft 온도를 정한다. 패치 점수 상위 5% 평균이 프레임 특징 점수다.
시간 점수는 최근 5개 예측 위상의 원형 변화량과 정상 fit 주기 길이 중앙값으로 계산한다. P3는 특징·시간 robust z를 0.5/0.5로 결합하고, P0–P2는 특징 z만 사용한다.
각 구성의 정상 검증 점수 99 percentile을 알람 임계값으로 고정한다. 모든 결과에 테스트 영상별 min-max 정규화는 사용하지 않는다.

```bash
python -m ipad_jepa.features --model dinov3-l --mode offline \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --upstream third_party/dinov3 --data-root "$IPAD_DATA_ROOT" \
  --devices R01 --splits test --batch-size 4 --workers 2

OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m ipad_jepa.experiment \
  --cache artifacts/features/dinov3-l/offline \
  --phase-run artifacts/runs/dinov3-l/offline/R01/seed0 \
  --data-root "$IPAD_DATA_ROOT" --device R01 --seed 0 \
  --out results/stage02/dinov3-l/offline/R01/seed0 \
  --local artifacts/runs/dinov3-l/offline/R01/seed0

python scripts/summarize_experiments.py
python scripts/plot_experiments.py
```

두 백본·세 seed 모두 실행한 뒤 요약/그림을 만든다. 메모리 NPZ는 `artifacts`에만 저장한다.
공개 근거는 실행별 `normal_fit.json`, `normal_calibration.csv`, P0–P3 프레임 점수 CSV, `metrics.json`, [device_summary.json](../results/stage02/device_summary.json)이다.
요약 도구는 공개 CSV에서 AUROC/AP를 다시 계산해 완료 메타데이터와 프레임 수·라벨 수·지표가 맞는지 확인한다. 모델 간 비교는 영상 ID·프레임 ID·라벨 일치를 검증한 뒤 같은 bootstrap 표본을 사용한다.
NumPy 기준 검색과 FP32 배치 검색의 hard/soft 및 전역/위상 결과 일치를 테스트했다. 나머지 장비, 온라인, LoRA, IPAD 기준선, 합성 진단, 실시간 측정과 추가 제거 실험이 완료될 때까지 전체 실험 완료를 선언하지 않는다.
