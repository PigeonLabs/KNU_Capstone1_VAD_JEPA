# Stage 02 — 고정 백본 특징 추출과 정상 위상 학습

## 완료 범위

DINOv3-L와 V-JEPA 2.1-L의 R01 오프라인 정상 학습 23개 영상과 정상 검증 5개 영상에서 특징을 추출했다.
학습은 4프레임 간격의 1,248개 clip, 검증은 1프레임 간격의 1,087개 clip이다. 테스트 영상과 진단 영상은 사용하지 않았다.
각 백본의 seed 0/1/2에서 20 epoch 위상 예측기 학습을 완료했다. PCA·메모리 구성과 이상탐지 평가, R02–R04 및 온라인 비교는 아직 수행 전이다. Stage 02 완료 태그는 생성하지 않는다.

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

seed 0 곡선은 대표 실행의 전체 epoch를 보여준다. 나머지 seed의 수치도 [phase_summary.csv](../results/stage02/phase_summary.csv)와 실행별 CSV/JSON에 공개한다. 이상탐지 AUROC/AP, 처리량, 알람 지연 결과는 아직 없다.

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
