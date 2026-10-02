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

## 완료된 정상 학습 — R01 오프라인 seed 0

두 백본 모두 20 epoch와 3,120회 optimizer 업데이트를 완료했다. 각 epoch에서 정상 fit 23개 영상의 1,248개 clip과 정상 calibration 5개 영상의 1,087개 clip을 사용했다. 아래 결과는 사전에 정한 **최소 정상 calibration CE**로 선택한 adapter와 joint head의 결과다. 원형 MAE는 주기 대비 백분율이며 선택 기준이 아니다.

| 백본 | 선택 epoch | 정상 calibration CE | 같은 epoch의 원형 MAE |
|---|---:|---:|---:|
| DINOv3-L | 18 | 4.7632 | 0.66% |
| V-JEPA 2.1-L | 1 | 5.3101 | 23.80% |

V-JEPA의 마지막 epoch 원형 MAE는 3.89%로 낮아졌지만 CE는 6.2001로 선택 epoch보다 높다. CE와 원형 MAE는 다른 지표이므로 선택 규칙을 바꾸지 않았고, 선택된 V-JEPA head의 위상 오차는 여전히 크다. 이 정상 학습 결과만으로 이상탐지 성능의 우열을 판단하지 않는다.

![DINOv3 정상 LoRA 학습 곡선](figures/stage04/dinov3-l_R01_offline_seed0_joint_training.png)

![V-JEPA 정상 LoRA 학습 곡선](figures/stage04/vjepa21-l_R01_offline_seed0_joint_training.png)

별표는 두 지표에서 동일한 CE 선택 epoch를 표시한다. teacher 곡선은 고정 teacher와의 정상 fit 특징 오차다. 이 그래프는 이상탐지 AUROC/AP나 실시간 FPS 측정 결과가 아니다. V-JEPA 학습은 앞서 기록한 저장공간 장애 이후 resume했으며, GPU 연속 학습과의 수치적 동일성을 입증하지 않는다.

완료된 [DINOv3 학습 CSV·JSON·검증 기록](../results/stage04/training/dinov3-l/offline/R01/seed0)과 [V-JEPA 학습 CSV·JSON·검증 기록](../results/stage04/training/vjepa21-l/offline/R01/seed0)을 공개한다. 기록에는 원본 곡선·metadata 해시와 선택 checkpoint 해시를 포함하며, checkpoint 파일은 업로드하지 않는다. [그래프 생성 코드](../scripts/plot_lora_training.py)는 20 epoch 완료와 최소 CE 선택을 검증한다.

두 백본의 R01 오프라인 seed 0은 선택된 adapter/head의 특징 재추출과 PCA·메모리·정상 보정 재구성, 15개 실제 테스트 영상 평가를 완료했다. P3 AUROC/AP는 DINOv3 **79.81/70.83%**, V-JEPA **33.06/27.52%**다. 같은 seed의 고정 백본 대비 DINOv3 P3의 AUROC/AP paired 차이 CI는 양수였고 V-JEPA는 0을 포함했다. 전역 P0의 AUROC는 두 백본 모두 낮아졌다. [single-seed 상세 결과·CI·그래프·출처 검증](LORA_R01_SEED0.md)을 제공하며, 아직 3-seed 평균이나 전체 장비 결과가 아니다.

R01 seed 1/2와 전체 장비·모드의 후속 학습·평가는 계속 진행 중이다.

## 전체 LoRA 행렬의 독립 검증과 집계

`scripts/summarize_lora_matrix.py`는 두 백본 × 두 모드 × 네 장비 × 세 seeds의 48조건을 검사한다. 실제 평가가 완료된 seed마다 20 epoch 정상 학습·최소 정상 CE 선택, 선택 adapter와 joint head의 tensor 일치, 실제 재추출 캐시의 메타데이터·target·차원, 재구성 PCA/메모리, 정상 보정과 실제 테스트 GT·점수·알람을 검증한다. 공개된 seed 0의 adapter·head·메모리 해시도 기존 검증 기록과 대조한다.

장비별 평균은 **같은 조건의 seeds 0/1/2가 모두 검증된 경우에만** 제공한다. seed별 AUROC/AP를 계산한 뒤 평균하며 seed 점수나 전체 장비 프레임을 합쳐 계산하지 않는다. 1,000회 원본 영상 bootstrap은 같은 장비·seed·영상의 고정 백본/LoRA, P0–P3 및 백본/모드 비교에 같은 추출을 사용한다. Macro4는 R01–R04를 같은 비중으로 평균하며 네 장비가 모두 있어야 생성한다.

```bash
PYTHONPATH=src:scripts python scripts/summarize_lora_matrix.py
# 전체 48조건 완료를 요구할 때만 추가:
PYTHONPATH=src:scripts python scripts/summarize_lora_matrix.py --require-full
# 독립 검증된 3-seed 그룹이 있어야 그래프를 생성한다.
PYTHONPATH=src:scripts python scripts/plot_lora_matrix.py --kind lora
```

검증 결과는 `results/stage04/matrix_audit`에 저장한다. seed 검증 기록과 집계 완료를 구분하며 아직 3-seed 그룹이 없으면 정확도 표·그래프를 만들지 않는다. P2/P3 정상 raw MAD·component q99와 모든 P0–P3의 정상 normalized q99, test GT·점수·연속 초과 알람을 재계산한다. P0/P1 정상 CSV에는 normalized 값만 있으므로 해당 raw median/MAD는 독립 재계산하지 않는다. 원본 영상의 encoder 재실행, PCA/k-center 재학습, GPU 특징 거리·온도 표본의 재계산 및 실시간 성능 측정은 이 CPU 검증의 범위에 포함하지 않는다.

현재 [독립 검증 기록](../results/stage04/matrix_audit/validation.json)은 DINOv3 R01 오프라인 seed 0, V-JEPA R01 오프라인 seeds 0/1의 **세 개 개별 조건**을 통과했다. LoRA와 고정 백본의 정상 임계값 총 24개를 확인했으며, 완성된 3-seed 그룹은 0개다. 신규 [V-JEPA seed 1의 실제 P0–P3 결과·학습 곡선](../results/stage04/vjepa21-l/offline/R01/seed1)은 P3 AUROC/AP **39.74/29.92%**다. 이 값은 단일 seed 결과이며 3-seed 평균·전체 Macro4·백본 우열을 의미하지 않는다. 세 개 seed 검증과 집계기는 구현 테스트를 포함한 전체 211개 테스트를 통과했으며, 실제 전체 행렬 완료와 구분한다.

teacher 가중치 0/1은 [별도 통제 실험](ABLATION_TEACHER_WEIGHT.md)에서 실제 두 joint LoRA 처리를 비교한다. 주 LoRA의 고정 백본 대비 차이를 teacher 0/1 결과로 해석하지 않는다.

전체 장비·3 seeds·2 modes의 LoRA 평가와 teacher weight 0/1을 포함한 추가 실험은 아직 완료되지 않았다. 실험 행렬은 [experiment_matrix.yaml](../configs/experiment_matrix.yaml), 예비 결과는 [results/stage04](../results/stage04)에 공개한다.


## 신규 DINOv3 R01 오프라인 seed 1과 네 조건 재검증

DINOv3 seed 1의 20 epoch·3,120회 업데이트와 선택 epoch **19**를 실제 checkpoint·정상 CE 곡선으로 확인했다. 같은 epoch 정상 위상 MAE는 **0.5977%**다. 실제 adapted 특징으로 PCA·프로토타입·보정을 다시 구성한 P3 AUROC/AP는 **83.55/76.47%**, 같은 seed 고정 백본 대비 **+39.86/+44.91pp**다. 1,000회 paired 영상 bootstrap의 95% CI는 각각 **+33.17..+48.42pp, +33.73..+55.12pp**로 0보다 높았다. 한 장비·한 시드 결과이며 세-seed 평균·전체 LoRA 우열을 뜻하지 않는다.

![DINOv3 seed 1의 정상 학습·선택 곡선](figures/stage04/dinov3-l_R01_offline_seed1_joint_training.png)

![DINOv3 seed 1의 실제 P3 비교·CI·ROC/PR](figures/stage04/dinov3-l_R01_offline_seed1_lora_evaluation.png)

[고정 영상 03의 점수·알람 그래프](figures/stage04/dinov3-l_R01_offline_seed1_lora_sequence03.png), [학습 export와 선택 해시](../results/stage04/training/dinov3-l/offline/R01/seed1), [실제 P0–P3 CSV·보정·single-seed CI·provenance](../results/stage04/dinov3-l/offline/R01/seed1)를 제공한다. 점수 그래프의 x축은 target index이며 실측 알람 지연이 아니다.

[current/retained 집계 검증](../results/stage04/retained_matrix_audit/validation.json)은 최초 두 백본 R01 오프라인 seeds 0/1, 네 조건·정상 임계값 32개를 확인했다. 아래 캐시 정리 후에는 현재 배열 세 조건·보존 증거 한 조건을 사용했다. V-JEPA seed 2 완료 시점에는 **다섯 조건·정상 임계값 40개·현재 배열 네 조건·보존 증거 한 조건**을 검증했다. 이후 DINOv3 seed 2를 반영한 현재 검증은 **여섯 조건·정상 임계값 48개·현재 배열 다섯 조건·보존 증거 한 조건·완결 세-seed 그룹 두 개**이며 아래 두 백본 비교로 이어진다. 기존 세 조건의 `matrix_audit` 증거를 보존하며 아래 새 결과의 범위를 구분한다. 전체 Macro4는 아직 없다.

```bash
PYTHONPATH=src:scripts python scripts/export_lora_training.py \
  --run artifacts/lora/dinov3-l/offline/R01/seed1 \
  --out results/stage04/training/dinov3-l/offline/R01/seed1
PYTHONPATH=src:scripts python scripts/plot_lora_training.py \
  --results results/stage04/training/dinov3-l/offline/R01/seed1 \
  --out docs/figures/stage04/dinov3-l_R01_offline_seed1_joint_training
```

[export_lora_training.py](../scripts/export_lora_training.py)는 실제 선택 checkpoint protocol·20개 epoch·최소 정상 CE와 원본 곡선을 확인하고 CSV/JSON·선택 해시만 export한다. 모델 텐서는 업로드하지 않는다. 이상탐지 비교 그래프는 기존 `plot_lora_evaluation.py`에 실제 training/cache/local/frozen-local 경로를 전달해 생성했다.

## 대형 adapted 특징 캐시의 보존·재검증 준비

전체 기본 LoRA와 teacher=0 비교의 FP16 특징을 계속 보관하면 추정 **약 1.28 TiB**가 필요하므로, 완료 조건의 결과 재검증 증거를 보존하는 경로를 추가했다. 고정 teacher 특징·원본 영상·선택 adapter/head·PCA/메모리·metadata/target·원본 점수 CSV는 보존 대상이다.

DINOv3 R01 오프라인 seed 0의 [실제 배열 inventory·원 감사 기록](../results/cache_retention/T1/dinov3-l/offline/R01/seed0)은 fresh 기존 감사 후 **43개 영상의 86개 NPY, 6,847,939,328 bytes(6.38 GiB)** 전체를 읽어 확인했다. 모든 FP16 값의 유한성, shape/dtype/contiguous layout, 전체 파일 SHA256·NPY header SHA256·파일 identity를 저장했다. **이 준비 명령은 파일을 삭제하지 않았으며, 현재 post-removal 검증 결과는 없다.**

[실제 retained tensor 사전 검증](../results/setup/retained_lora_tensor_preflight.json)은 완료된 네 조건의 현재 선택 adapter·joint head 동등성·선택 epoch·재구성 메모리·고정 teacher 출처를 retained helper로 확인했다. 이 증거는 현재 배열 감사가 존재하는 상태의 사전 검증이다. 전체 encoder 재추출이나 캐시 제거 후 재현을 입증하지 않는다.

```bash
PYTHONPATH=src:scripts python scripts/prepare_lora_cache_retention.py \
  --model dinov3-l --mode offline --device R01 --seed 0 \
  --data-root "$IPAD_DATA_ROOT"
PYTHONPATH=src:scripts python scripts/summarize_retained_lora_matrix.py \
  --kind lora --data-root "$IPAD_DATA_ROOT"
PYTHONPATH=src:scripts python scripts/check_retained_lora_tensors.py
# 모든 장비·세 seeds가 완료된 최종 검증:
PYTHONPATH=src:scripts python scripts/summarize_retained_lora_matrix.py \
  --kind lora --require-full --data-root "$IPAD_DATA_ROOT"
PYTHONPATH=src:scripts python scripts/plot_retained_lora.py \
  --root results/stage04/retained_matrix_audit \
  --data-root "$IPAD_DATA_ROOT" --out docs/figures/stage04/retained
```

새 집계기는 현재 payload가 있으면 기존 실제 배열 감사로 검증한다. payload가 없으면 완결된 의도적 제거 journal과 정확한 성공 CI revision에 archive된 inventory·원 감사가 있어야 retained replay를 허용한다. 제거 journal이 중간 상태이거나 파일이 임의로 없어진 조건은 통과시키지 않는다. retained 경로에서는 원 payload 유한값/해시 검사가 **과거 증거**임을 표시하고 현재 metadata/target·선택 텐서·보정·GT·시간·점수·알람을 다시 검사한다.

같은 seeds 0/1/2가 모두 검증된 그룹만 평균·paired CI·그림을 만들고, 네 장비가 모두 있어야 동일 가중치 Macro4를 제공한다. teacher=0/1 경로는 두 실제 학습 조건과 일치하는 통제를 요구한다. 현 시점 teacher=0/1 완료 쌍이나 캐시 제거 후 검증은 없다. 최종 조건 전체의 GPU 평가, 안전한 실제 제거 실행·제거 후 감사와 최종 집계/그래프는 남아 있다.


## 캐시 정리 실행 코드와 중단 재개 검증

[retire_lora_cache.py](../scripts/retire_lora_cache.py)와 [cache_retirement.py](../src/ipad_jepa/cache_retirement.py)는 archive된 inventory에 열거된 완료 조건의 `patch.npy`·`global.npy`만 처리한다. 원본·teacher 특징·선택 모델·head·PCA/메모리·metadata/target·평가 CSV를 삭제 대상에 넣으면 거부한다. 같은 조건의 kernel lock과 현재 실제 GPU 비사용, live runtime controller의 측정 사이 hold를 확인한다.

준비 모드는 정확한 inventory 공개 revision의 **CPU invariants and upload guard 성공 CI**를 실제 GitHub에서 확인한다. 현재 metadata/target·선택 텐서·정상 보정·GT/P3와 86개 원 파일의 전체 SHA256·NPY header·파일 identity를 재검증하고 [삭제 없는 구체적인 계획](../results/cache_retention/T1/dinov3-l/offline/R01/seed0/retirement_plan.json)을 저장했다. 실제 삭제는 구현 코드와 이 계획이 같은 성공 CI revision에 공개된 후 별도 `--execute`로만 진행한다.

실행은 모든 남은 파일을 먼저 검증한 뒤 각 파일을 삭제 직전에 다시 전체 해시 검증한다. fsync한 journal에 unlink 의도를 먼저 저장하고, 실제 파일 부재를 확인한 뒤 완료 항목을 저장한다. 중단 후 재개는 정확한 같은 조건·source·plan, 이전 owner의 실제 종료, 완료 prefix와 다음 파일의 durable intent를 확인한다. 기록 없는 누락·완료 파일의 재생성·활성 owner·symlink 교체·다른 성공 workflow는 통과시키지 않는다. 완료된 삭제 journal과 이후 retained 과학적 감사는 별도 증거로 저장한다.

27개 신규 파일시스템/CLI 검증을 포함한 전체 **291개 로컬 테스트**를 통과했다. 실제 파일 삭제 전후의 중단 창, size/inode/mtime를 보존한 payload 변조, 누락·재생성·경로 이탈과 잘못된 CI/실시간 동시 작업을 확인했다. 이 테스트는 fixture 기반이며 실제 모델 조건의 post-removal 감사는 아직 없다.

```bash
# 기본은 삭제 없는 prepare:
PYTHONPATH=src:scripts python scripts/retire_lora_cache.py \
  --receipt results/cache_retention/T1/dinov3-l/offline/R01/seed0/payload_inventory.json \
  --publication-revision 8e8abbb323ad5e24eb8fec1ec9c5c71a6533f69a \
  --publication-ci-id 36984566442 --data-root "$IPAD_DATA_ROOT"
# 위 계획/구현을 공개한 정확한 성공 revision과 CI ID를 지정한 뒤 실행:
PYTHONPATH=src:scripts python scripts/retire_lora_cache.py \
  --receipt results/cache_retention/T1/dinov3-l/offline/R01/seed0/payload_inventory.json \
  --publication-revision 8e8abbb323ad5e24eb8fec1ec9c5c71a6533f69a \
  --publication-ci-id 36984566442 \
  --implementation-revision "$RETIRE_IMPLEMENTATION_REV" \
  --implementation-ci-id "$RETIRE_IMPLEMENTATION_CI" \
  --data-root "$IPAD_DATA_ROOT" --execute
# 실제 중단 journal이 있을 때만 같은 인자에 --resume 추가.
```

## 첫 실제 캐시 정리와 사후 재검증

사용자가 승인한 조건은 **후속 실험에서 사용하지 않는 재생성 가능한 특징 배열만 삭제**하는 것이다. [의존성·재생성 입력 검증](../results/cache_retention/T1/dinov3-l/offline/R01/seed0/dependency_regeneration_check.json)은 해당 조건의 원본 43개 영상 전체 해시, 기본 가중치·upstream revision·선택 adapter·고정 소스를 실제 확인하고 CPU 런타임 adapter 바인딩을 통과했다. 실시간 측정은 원본과 선택 모델/head/bank를 읽고, teacher=0 학습은 별도 고정 teacher 캐시를 읽는다. 진단은 고정 백본과 held-out 정상 원본을 사용하며, LoRA·teacher 집계는 current/retained 명령으로 검증한다. GPU로 전체 특징을 재생성하거나 bitwise 일치를 검증한 것은 아니다. 필요하면 같은 원본·선택 모델·고정 소스로 별도의 새 캐시 경로에 재추출할 수 있다.

실행은 inventory revision `8e8abbb323ad5e24eb8fec1ec9c5c71a6533f69a`의 성공 CI와 구현/계획 revision `533a08da7819d75025852d00d0dc0f7296f2a25f`의 [성공 CI](https://github.com/PigeonLabs/KNU_Capstone1_VAD_JEPA/actions/runs/36989545978)를 실제 확인했다. 최신 구현 CI는 외부 `PYTHONPATH` 없이 **291개 테스트와 전체 업로드 검사**를 통과했다.

| DINOv3-L LoRA / offline / R01 / seed 0 | 정리 전 archive | 정리 후 실제 확인 |
|---|---:|---:|
| patch/global 특징 배열 | 86개 / 6,847,939,328 bytes (6.38 GiB) | 0개 / 0 bytes |
| metadata JSON | 43개 | 43개, 원 해시 일치 |
| target NPY | 43개 | 43개, 원 해시 일치 |

[실제 제거 journal](../results/cache_retention/T1/dinov3-l/offline/R01/seed0/retirement.json), [사후 독립 감사](../results/cache_retention/T1/dinov3-l/offline/R01/seed0/post_retirement_audit.json), [실행 결과](../results/cache_retention/T1/dinov3-l/offline/R01/seed0/retirement_execution_check.json)를 제공한다. 원본·teacher·선택 adapter·joint head·PCA/메모리·평가 CSV를 보존했고, 현재 텐서·metadata/target·정상 임계값·GT·위상/시간/점수/알람/지표를 재검증했다. 전체 네 LoRA 조건의 정상 임계값 32개를 다시 검증했으며, 정리한 한 조건은 archived/current replay를 사용했다. 특징 배열의 원 해시·유한값 검사는 정리 전의 **과거 증거**이고 encoder 재추출·PCA 재학습·GPU 검색 재실행은 이 사후 검증 범위에 포함하지 않는다.

![실제 특징 배열 정리와 metadata/target 보존](figures/stage04/retained/dinov3_l_offline_R01_seed0_cache_retirement.png)

[그래프 코드](../scripts/plot_cache_retirement.py)는 현재 파일 부재·원 metadata/target 해시·완료 journal·사후 감사 일치를 재확인하고 PNG/SVG와 [수치/출처 receipt](figures/stage04/retained/dinov3_l_offline_R01_seed0_cache_retirement.json)를 생성한다. 단일 완료 조건의 논리적 파일 용량이며 전체 디스크 절약량이나 탐지 성능 개선을 뜻하지 않는다.

## 중단된 주 LoRA 조건 이어가기

[continue_lora_condition.py](../scripts/continue_lora_condition.py)는 기본 teacher=1의 한 조건을 기존 **20 epoch 정상-only 프로토콜**로 이어간다. 기존 원 trainer가 checkpoint의 protocol·source·optimizer/RNG를 확인하고, 전체 학습 완료 후 정상 검증 CE로 선택된 모델의 fit/calibration/test 특징, 새 PCA/메모리·정상 보정, 전체 실제 테스트 평가, 독립 조건 감사를 순차 수행한다. 한 조건의 완료가 전체 48조건의 완료를 뜻하지 않는다.

```bash
PYTHONPATH=src python scripts/continue_lora_condition.py \
  --data-root "$IPAD_DATA_ROOT" --model vjepa21-l \
  --mode offline --device R01 --seed 2 --resume
```

실제 사용 시 CUDA 학습 환경의 Python으로 실행한다. 단일 accuracy lock과 live runtime controller의 측정 사이 hold, GPU 비사용을 확인한 뒤 child를 시작하고 실제 PID/boot/start identity·종료 코드·고정 source 해시를 기록한다. shortened epoch·pilot·다른 teacher/seed·checkpoint 없는 재개·활성 owner의 중복 실행은 거부한다. 8개 신규 continuation 검증을 포함한 전체 **299개 로컬 테스트**를 외부 `PYTHONPATH` 없이 통과했다. 실제 GPU 재개와 최종 정확도는 별도 완료 증거로 구분한다.

V-JEPA 2.1-L R01 오프라인 seed 2는 기존 1 epoch·156 updates의 체크포인트에서 실제 재개했다. [시작 시점의 host owner/child·protocol/source 확인](../results/setup/lora_seed2_continuation_start_check.json)은 GPU 학습 child와 실시간 실행기의 hold를 확인한 snapshot이다. 시작 snapshot과 아래 실제 학습·평가 완료 근거를 구분한다.

## V-JEPA 2.1-L R01 오프라인 seed 2의 완료된 학습

기존 checkpoint에서 재개한 원 trainer가 **20 epoch·3,120 updates**를 완료하고 실제 종료 코드 0을 반환했다. 정상 fit 23개 영상의 1,248 clips, 정상 calibration 5개 영상의 1,087 clips, teacher weight 1·accumulation 8과 원 학습 코드를 유지했다. 실제 선택 checkpoint의 protocol과 아래 20개 epoch 곡선을 검증했으며, 최소 정상 calibration CE로 epoch **18**을 선택했다.

| 선택 epoch | 정상 calibration CE | 같은 epoch의 원형 MAE |
|---:|---:|---:|
| 18 | 5.250174 | 1.5681% |

![V-JEPA seed 2의 완료된 정상 학습·CE 선택](figures/stage04/vjepa21-l_R01_offline_seed2_joint_training.png)

[학습 CSV·JSON·선택 checkpoint 해시·export 검증](../results/stage04/training/vjepa21-l/offline/R01/seed2)을 제공한다. 그래프의 별표는 CE와 MAE 패널에서 같은 선택 epoch를 표시한다. MAE 최솟값으로 모델을 다시 선택하지 않았다. 모델 파일은 업로드하지 않는다.

위 곡선은 **정상 학습**의 결과이며 AUROC/AP나 실시간 FPS를 나타내지 않는다. [특징 재추출 당시의 학습 단계 확인·소스/그래프 해시](../results/setup/lora_vjepa21_seed2_training_publication_check.json)는 학습 완료를 검증한 snapshot이다. 이후 전체 파이프라인도 실제 완료했으며 그 정확도는 아래에서 보고한다. 재개 학습이 중단 없는 GPU 학습과 bitwise 같다는 검증은 아직 없다.

## V-JEPA R01 오프라인 첫 세-seed LoRA 비교

Seed 2는 선택 epoch 18의 실제 fit/calibration/test 특징을 재추출하고, 새 PCA·프로토타입·정상 보정을 구성해 **15개 실제 테스트 영상**을 평가했다. 공통 GT target 3,295개 중 이상 target은 1,227개다. 원 trainer·재추출·전체 평가 child의 종료 코드 0, 선택 adapter/joint head·캐시 출처·메모리·정상 임계값·GT·시간/점수/알람·지표의 독립 검증과 실제 부모 프로세스의 종료를 확인했다.

Seed 2 P3 AUROC/AP는 **41.13/31.85%**, 같은 seed 고정 백본 대비 **+2.21/+2.61pp**다. 1,000회 paired 영상 bootstrap의 95% CI는 각각 **−4.04..+8.51pp, −0.79..+7.51pp**로 0을 포함한다.

![V-JEPA seed 2의 실제 고정 백본 비교·ROC/PR](figures/stage04/vjepa21-l_R01_offline_seed2_lora_evaluation.png)

[고정 영상 03의 점수/알람 궤적](figures/stage04/vjepa21-l_R01_offline_seed2_lora_sequence03.png), [P0–P3 전체 CSV·정상 보정·개별 paired CI·provenance](../results/stage04/vjepa21-l/offline/R01/seed2), [원 파이프라인 완료 증거](../results/stage04/condition_pipeline_checks/vjepa21-l/offline/R01/seed2/pipeline_completion.json)를 공개한다. 궤적의 x축은 target index이며 실측 알람 지연이 아니다.

동일 조건의 **seeds 0/1/2를 모두 검증**한 첫 LoRA 그룹은 다음과 같다. 각 seed의 지표를 계산한 뒤 평균하고, 두 처리·세 seeds에 같은 원본 영상 재표집을 적용한 1,000회 paired bootstrap의 percentile 95% CI를 구했다.

| 처리 | AUROC 평균 (%) | AP 평균 (%) |
|---|---:|---:|
| 고정 백본 P3 | 37.58 | 28.83 |
| LoRA P0 | 35.01 | 29.22 |
| LoRA P3 | 37.98 | 29.76 |

LoRA P3−고정 P3의 차이는 **AUROC +0.40pp [−2.11, +2.92], AP +0.93pp [−0.37, +2.79]**다. 두 CI 모두 0을 포함해 이 장비·모드에서 개선이 확인되지 않았다. seed 2의 정상 위상 MAE가 작다는 사실만으로 이상탐지 성능 개선을 주장하지 않는다.

![첫 V-JEPA 세-seed LoRA 평균과 paired 차이](figures/stage04/retained/retained_lora_vjepa21-l_offline_R01.png)

[현재 여섯 조건·48개 정상 임계값 검증](../results/stage04/retained_matrix_audit/validation.json), [모든 P0–P3 평균·CI·paired 차이](../results/stage04/retained_matrix_audit/device_summary.json), [새 조건·그룹의 공개 확인](../results/setup/lora_vjepa21_seed2_evaluation_publication_check.json)을 제공한다. 이 V-JEPA 완료 snapshot에서는 **5/48조건·1/16그룹**이었다. 이후 DINOv3 seed 2를 포함한 현재 상태는 **6/48조건·2/16그룹**이며 아래 두 백본 비교를 추가했다. 이 CI는 고정된 세 seeds의 영상 재표집이며 seed 자체를 재표집하지 않았다. 다른 장비·온라인 LoRA·전체 Macro4·실시간 비교는 남아 있다.


## DINOv3 R01 오프라인 seed 2의 완료된 정상 학습

기본 teacher weight 1·accumulation 8·BF16의 원 정상-only trainer로 **20 epoch·3,120 updates**를 완료했다. 정상 fit 23개 영상의 1,248 clips와 calibration 5개 영상의 1,087 clips를 사용했다. trainer child의 종료 코드 0과 완료 metadata·20개 epoch CSV·선택 checkpoint protocol을 확인했다. 최소 정상 calibration CE로 epoch **11**을 선택했다.

| 선택 epoch | 정상 calibration CE | 같은 epoch의 원형 MAE |
|---:|---:|---:|
| 11 | 4.336612 | 1.0314% |

![DINOv3 seed 2의 완료된 정상 학습·CE 선택](figures/stage04/dinov3-l_R01_offline_seed2_joint_training.png)

[실제 학습 CSV·JSON·선택 checkpoint 해시·export 검증](../results/stage04/training/dinov3-l/offline/R01/seed2)을 제공한다. 그래프의 두 별표는 같은 선택 epoch를 표시하며 MAE 최솟값으로 재선택하지 않았다. 모델 텐서는 업로드하지 않는다. **정상 학습 곡선은 AUROC/AP나 FPS 결과가 아니다.** [학습 단계의 소스·검증·선택 특징 재추출 시작 snapshot](../results/setup/lora_dinov3_seed2_training_publication_check.json)과 이후 전체 평가 완료 증거를 구분한다. 이 학습 단계 snapshot에서 독립 검증된 정확도는 여전히 5/48조건이며 DINOv3의 세-seed 그룹은 아직 없다.

선택 epoch의 fit/calibration/test 특징 재추출·새 PCA/메모리·정상 보정·전체 실제 테스트 평가·독립 감사를 기존 파이프라인에서 이어간다. epoch·loss·데이터 분할·점수 부호·정상 임계값을 테스트 결과에 맞춰 바꾸지 않는다.

### 완결된 LoRA 백본 쌍의 비교 그래프

[plot_lora_backbone_comparison.py](../scripts/plot_lora_backbone_comparison.py)는 같은 장비·모드에서 **DINOv3와 V-JEPA 각각 seeds 0/1/2가 모두 검증된 그룹**만 사용한다. frozen P3·LoRA P0·LoRA P3의 평균·CI와 V-JEPA−DINOv3 paired 차이를 실제 CSV에서 다시 계산한다. current/retained 조건 감사도 새로 수행하며 정리된 배열의 원 해시·유한성 검사는 과거 증거임을 유지한다.

부분 seed·서로 다른 장비/모드·GT/target 불일치·잘못된 paired 방향·틀린 요약 수치는 거부한다. 네 장비가 모두 있어야 Macro4를 제공하고, seed 지표의 평균과 영상 재표집을 사용한다. seed 자체나 pooled seed 점수는 재표집하지 않는다. 새 9개 비교 검사를 포함한 전체 CPU 테스트 **364개**를 통과했다. 이 CPU 검증 자체는 실제 완결된 백본 쌍의 결과를 뜻하지 않는다. 학습 단계 snapshot의 한 완결 그룹만으로 비교 그림을 만드는 실제 CLI 요청은 거부했다. 이후 실제 DINOv3 전체 평가가 완료돼 아래 완결된 쌍의 그래프를 생성하고 원 수치를 재검산했다.

```bash
# 두 백본의 같은 조건 세 seeds가 모두 실제 평가·검증된 후:
PYTHONPATH=src:scripts python scripts/summarize_retained_lora_matrix.py \
  --kind lora --data-root "$IPAD_DATA_ROOT"
PYTHONPATH=src:scripts python scripts/plot_lora_backbone_comparison.py \
  --root results/stage04/retained_matrix_audit \
  --data-root "$IPAD_DATA_ROOT" --out docs/figures/stage04/retained
```


## DINOv3 R01 오프라인 seed 2의 완료된 전체 평가

선택 epoch 11의 실제 특징을 **43개 fit/calibration/test 영상**에서 재추출했다. 선택 adapter/joint head를 유지하고 PCA·위상별 프로토타입·온도·정상 median/MAD/q99를 새로 구성해 **15개 실제 테스트 영상**을 평가했다. 원 trainer·재추출·메모리/전체 평가 child의 종료 코드 0과 부모 프로세스 종료, 독립 조건 감사 통과를 확인했다. 공통 GT target은 3,295개이며 이상 target은 1,227개다.

| seed 2 처리 | AUROC (%) | AP (%) |
|---|---:|---:|
| LoRA P0 | 66.55 | 48.03 |
| LoRA P1 | 63.03 | 49.38 |
| LoRA P2 | 65.22 | 51.27 |
| LoRA P3 | 77.28 | 67.29 |

같은 seed 고정 P3 대비 차이는 **AUROC +28.70pp [20.75, 35.94], AP +33.18pp [18.26, 43.57]**다. 단일 seed의 1,000회 paired 원본 영상 bootstrap CI이며 학습 seed 불확실성이나 Macro4를 뜻하지 않는다. P0는 같은 seed의 고정 P0 대비 **−13.69/−10.86pp**로 두 CI가 음수였다. 이 정상 학습이 모든 점수 구성에 일관된 개선을 준 결과는 아니다.

![DINOv3 seed 2의 실제 P3 비교·CI·ROC/PR](figures/stage04/dinov3-l_R01_offline_seed2_lora_evaluation.png)

[고정 영상 03의 점수/알람 궤적](figures/stage04/dinov3-l_R01_offline_seed2_lora_sequence03.png), [P0–P3 전체 CSV·보정·single-seed CI·provenance](../results/stage04/dinov3-l/offline/R01/seed2), [원 파이프라인 완료 증거](../results/stage04/condition_pipeline_checks/dinov3-l/offline/R01/seed2/pipeline_completion.json)를 제공한다. 궤적의 target index는 실제 알람 wall-clock 지연이 아니다. 실제 재생 속도와 이벤트 지연은 Stage 05에서 별도로 측정한다.

## R01 오프라인 첫 두 백본 세-seed LoRA 비교

두 백본 각각 **seeds 0/1/2를 모두 검증**했다. 같은 15개 영상·3,295개 GT target에서 각 seed의 지표를 먼저 계산해 평균했다. 세 seeds와 두 백본/처리에 동일한 원본 영상 재표집을 적용한 1,000회 paired bootstrap의 percentile 95% CI이며 seed 자체는 재표집하지 않았다.

| 백본 / 처리 | AUROC 평균 (%) | AP 평균 (%) |
|---|---:|---:|
| DINOv3 고정 P3 | 47.01 | 33.21 |
| DINOv3 LoRA P0 | 71.36 | 54.16 |
| DINOv3 LoRA P3 | 80.21 | 71.53 |
| V-JEPA 고정 P3 | 37.58 | 28.83 |
| V-JEPA LoRA P0 | 35.01 | 29.22 |
| V-JEPA LoRA P3 | 37.98 | 29.76 |

| paired 차이 | AUROC 차이 / 95% CI (pp) | AP 차이 / 95% CI (pp) |
|---|---:|---:|
| DINOv3 LoRA P3−고정 P3 | +33.20 [28.24, 39.72] | +38.32 [26.28, 47.47] |
| V-JEPA LoRA P3−고정 P3 | +0.40 [−2.11, 2.92] | +0.93 [−0.37, 2.79] |
| LoRA P3 V-JEPA−DINOv3 | −42.24 [−51.30, −33.33] | −41.76 [−50.26, −29.44] |

이 장비·모드에서 DINOv3 LoRA P3 개선과 두 LoRA 백본의 차이가 확인됐다. V-JEPA의 고정 P3 대비 개선 CI는 두 지표 모두 0을 포함했다. 별도 정상-only joint head·adapted encoder·재구성 메모리/보정까지 포함한 두 프로토콜의 비교다. 특정 encoder나 teacher 손실 하나의 단독 원인, 다른 장비·온라인의 우열, 전체 LoRA Macro4를 입증하지 않는다. teacher 0/1 통제 결과도 아직 없다.

![DINOv3 R01 세-seed LoRA 평균과 paired 개선](figures/stage04/retained/retained_lora_dinov3-l_offline_R01.png)

![두 백본의 세-seed LoRA 평균과 paired 백본 차이](figures/stage04/retained/lora_backbone_offline_R01.png)

[여섯 조건·48개 정상 임계값·두 세-seed 그룹 감사](../results/stage04/retained_matrix_audit/validation.json), [모든 F/L P0–P3 평균·CI와 18개 paired 비교](../results/stage04/retained_matrix_audit/device_summary.json), [fresh current/retained 그룹 그림 근거](../results/stage04/retained_matrix_audit/figure_sources.json), [실제 CSV에서 재검산한 백본 그림 수치·출처](../results/stage04/retained_matrix_audit/backbone_figure_sources.json)를 제공한다. 원 DINOv3 seed 0 배열은 의도적 정리 전 archive와 현재 보존 자료로 재검증했고 나머지 다섯 조건은 현재 실제 adapted 배열 증거를 사용했다. **전체 6/48조건·2/16그룹**이며 전체 Macro4는 아직 없다.

다음 주 LoRA 조건은 **DINOv3-L 오프라인 R02 seed 0**이다. 기존 canonical 출력이 없는 새 조건과 전체 adapted 캐시+64 GiB 여유 공간을 확인하고 원래 20 epoch·teacher 1 프로토콜로 시작했다. [이번 완료 조건·그룹·그래프와 다음 실제 child 확인](../results/setup/lora_dinov3_seed2_evaluation_publication_check.json)을 제공한다. R02/12·13·14의 라벨 정렬은 계속 미확정이며 기존 공통 제외 mask와 오프셋 민감도 규칙을 유지한다.


## DINOv3 R02 오프라인 seed 0의 완료된 정상 학습

원래 teacher weight 1·accumulation 8·BF16 프로토콜로 **20 epoch·7,620 optimizer updates**를 완료했다. epoch마다 정상 fit **3,042 clips**, 정상 calibration **2,864 clips**를 사용했다. 최소 정상 calibration CE에 따라 epoch **13**을 선택했으며, 이상 라벨은 선택에 사용하지 않았다. 완료 metadata·20개 epoch CSV·선택 checkpoint protocol과 학습 child 종료 코드 0을 확인했다.

| 선택 epoch | 정상 calibration CE | 같은 epoch의 원형 MAE |
|---:|---:|---:|
| 13 | 3.321761 | 1.9929% |

![DINOv3 R02 seed 0의 완료된 정상 학습·CE 선택](figures/stage04/dinov3-l_R02_offline_seed0_joint_training.png)

[학습 CSV·JSON·선택 checkpoint 해시·export 검증](../results/stage04/training/dinov3-l/offline/R02/seed0)을 제공한다. 두 별표는 같은 CE 선택 epoch이며 MAE로 재선택하지 않았다. teacher 곡선은 고정 teacher와의 정상 fit 특징 오차다. 이 곡선은 이상탐지 AUROC/AP나 실시간 FPS 결과가 아니다. 모델 텐서는 업로드하지 않는다.

사용자 요청에 따라 20 epoch 이후 작업을 일시정지했고, 명시적인 재개 요청을 받은 뒤 **완료 모델을 유지한 채** 선택 특징 재추출·새 PCA/프로토타입 메모리·정상 보정·모든 실제 테스트 영상 평가를 이어간다. 원 실행 보호 조건을 유지하기 위해 종료된 실시간 제어기도 원래 resume 기능으로 복구해 측정 대기 상태로 두었다. [완료 학습·재개 단계 검증 snapshot](../results/setup/lora_dinov3_R02_seed0_training_publication_check.json)을 제공한다. 이 학습 단계 snapshot에서 독립 검증된 LoRA 정확도는 **6/48조건**이며, R02 정확도·세-seed 평균·Macro4는 아직 없다. R02/12·13·14의 라벨 정렬은 계속 미확정이므로 기존 공통 제외 mask와 오프셋 민감도 규칙을 유지한다.

```bash
PYTHONPATH=src:scripts python scripts/export_lora_training.py \
  --run artifacts/lora/dinov3-l/offline/R02/seed0 \
  --out results/stage04/training/dinov3-l/offline/R02/seed0
PYTHONPATH=src:scripts python scripts/plot_lora_training.py \
  --results results/stage04/training/dinov3-l/offline/R02/seed0 \
  --out docs/figures/stage04/dinov3-l_R02_offline_seed0_joint_training
```
