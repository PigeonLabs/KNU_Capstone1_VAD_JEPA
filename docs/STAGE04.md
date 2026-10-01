# Stage 04 — 정상 데이터 기반 LoRA와 추가 실험

## 구현과 검증 범위

두 공식 ViT-L 백본의 마지막 4개 attention block에 q/v LoRA를 적용한다. rank 8, alpha 16, dropout 0.05이며, encoder 학습 파라미터는 131,072개다. 313,800개 파라미터의 위상 head를 함께 학습한다. decoder나 V-JEPA predictor를 사용하지 않는다.

DINOv3 attention이 읽는 fused projection의 `in_features` 속성을 유지하는 별도 래퍼를 사용한다. 원본 가중치와 k projection은 고정한다. encoder는 eval 상태를 유지하고 LoRA dropout과 phase head만 train 상태로 전환해 고정 prefix의 drop-path/RoPE 증강이 추가되지 않게 한다.

정상 fit 영상의 raw clip과 검증된 고정 encoder의 dense 특징을 짝지어 학습한다. teacher cache의 영상·프레임·모델·입력 모드·코드 해시와 원본 프레임 내용을 확인한다. 진단 영상과 실제 테스트 영상을 학습·선택에 사용하지 않는다.

손실은 `phase CE + teacher_weight × dense L2`다. dense L2는 각 patch의 1,024차원 특징을 channel 방향으로 단위 정규화한 뒤, 차원별 제곱 오차를 합산하고 patch·batch에 대해 평균한다. teacher는 gradient를 받지 않는다. 차원 수로 추가 나누지 않는다.

AdamW의 encoder LR은 1e-4, head LR은 1e-3, weight decay는 1e-4다. BF16, micro-batch 1, 8회 gradient accumulation, 20 epoch를 사용한다. 마지막 불완전 accumulation window도 실제 clip 수로 나누고 반영한다. 정상 calibration phase CE가 가장 낮은 epoch를 선택하며, 원형 MAE는 함께 보고한다.

## 실행 경로

```bash
python -m ipad_jepa.train_lora --model dinov3-l --mode offline --device R01 \
  --data-root "$IPAD_DATA_ROOT" --teacher-cache artifacts/features/dinov3-l/offline \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --upstream third_party/dinov3 --seed 0 \
  --out artifacts/lora/dinov3-l/offline/R01/seed0

python -m ipad_jepa.adapted_features \
  --run artifacts/lora/dinov3-l/offline/R01/seed0 \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --upstream third_party/dinov3 --data-root "$IPAD_DATA_ROOT" \
  --cache artifacts/features_lora/dinov3-l/offline/R01/seed0 \
  --phase-out artifacts/runs_lora/dinov3-l/offline/R01/seed0

python -m ipad_jepa.experiment --device R01 --seed 0 \
  --cache artifacts/features_lora/dinov3-l/offline/R01/seed0 \
  --phase-run artifacts/runs_lora/dinov3-l/offline/R01/seed0 \
  --data-root "$IPAD_DATA_ROOT" \
  --out results/stage04/dinov3-l/offline/R01/seed0 \
  --local artifacts/runs_lora/dinov3-l/offline/R01/seed0
```

V-JEPA와 온라인에도 같은 조건을 적용한다. 선택된 adapter와 동시에 선택된 head를 사용하며 head를 후속 단계에서 별도 재학습하지 않는다. 특징 추출 후 정상 fit/calibration 경로로 PCA·프로토타입·temperature·score calibration을 모두 다시 구성한다. 대용량 파일은 로컬에만 저장한다.

각 epoch 종료 시 adapter/head, optimizer, Python/NumPy/CPU/CUDA/loader RNG와 곡선을 저장한다. `--resume`은 동일 코드·teacher fingerprint·설정만 허용하고 완료 epoch부터 재개한다. GPU resume의 연속 실행과의 일치는 아직 검증하지 않았다.

`--max-steps`는 실제 optimizer 업데이트를 확인하는 pilot 전용이다. pilot과 전체 학습을 구분하며, 20 epoch가 완료되지 않은 adapter를 특징 재추출·주 평가에 전달하면 거부한다. GPU pilot은 학습 연산의 증거이며 이상탐지 성능이나 실시간 FPS의 증거는 아니다.

## 실제 GPU 예비 학습

R01 정상 fit에서 각 백본의 8개 micro-batch를 누적한 optimizer 업데이트 1회를 실행했다. 두 실행 모두 131,072개 encoder 파라미터와 313,800개 head 파라미터를 학습했다.

| 백본 | 정상 clip | optimizer 업데이트 | 평균 phase CE | 평균 normalized dense L2 |
|---|---:|---:|---:|---:|
| DINOv3-L | 8 | 1 | 5.3008 | 3.3923e-5 |
| V-JEPA 2.1-L | 8 | 1 | 5.1523 | 4.1493e-8 |

이 값은 초기 8개 학습 clip의 손실이며 학습 완료나 모델의 우열을 뜻하지 않는다. [예비 결과 JSON](../results/stage04/pilots)에 source/teacher fingerprint와 학습 후 q/v B 행렬 norm을 공개한다. 백본별 배치 구성과 특징 정밀도 차이를 포함하므로 초기 teacher loss를 이상탐지 성능으로 비교하지 않는다.

두 백본의 R01 오프라인 20 epoch·3-seed 학습과 후속 특징·메모리 평가를 실행 중이다. 선택된 LoRA 특징과 재구성된 PCA·메모리의 실제 테스트 평가는 아직 완료되지 않았다.

전체 장비·3 seeds·2 modes의 LoRA 평가와 teacher weight 0/1을 포함한 추가 실험은 아직 완료되지 않았다. 실험 행렬은 [experiment_matrix.yaml](../configs/experiment_matrix.yaml), 예비 결과는 [results/stage04](../results/stage04)에 공개한다.
