# IPAD × DINOv3 / V-JEPA 2.1

정상 공정 영상으로 위상별 프로토타입 메모리를 구성하고, **디코더 없이 특징 잔차와 시간적 위상 이탈**로 이상을 탐지한다.
IPAD 실제 장비 R01–R04에서 두 백본의 **온라인·오프라인** 정확도와 실제 알람 지연을 비교한다.

`16프레임 → 백본 특징 → 위상 예측 → 정상 메모리 검색 → 특징·시간 점수 → 알람`

## 현재 공개 결과

**고정 백본의 두 모델 × 두 모드 × 네 장비 비교를 3 seeds로 완료했다. LoRA·IPAD 전체 기준선·추가 실험·실시간 측정은 진행 중이다.**
학습 111개 영상/50,642프레임, 실제 테스트 66개 영상/33,462프레임을 확인했다. 정상 분할은 fit 76개·검증 18개·진단 17개다.
R02/12·13·14는 정확한 라벨 정렬이 미확정이다. ±1 인덱스 가정에 따른 18프레임 제외와 오프셋 민감도를 [Stage 00](docs/STAGE00.md)에 기록했다. 고정 백본 비교는 같은 31,728개 유효 테스트 프레임을 사용한다.

![정상 영상 분할](docs/figures/stage00/normal_split.png)

네 장비 동일 가중치 Macro4이며, 각 장비의 seed 0/1/2 지표를 먼저 평균했다. 아래는 주 방법 P3의 결과다.

| 백본 | 모드 | P3 Macro AUROC / AP (%) | p95 전체 지연 | 지속 FPS |
|---|---|---:|---:|---:|
| DINOv3-L | 오프라인 | 64.15 / 55.10 | — | — |
| DINOv3-L | 온라인 | 66.37 / 55.92 | — | — |
| V-JEPA 2.1-L | 오프라인 | 55.59 / 47.14 | — | — |
| V-JEPA 2.1-L | 온라인 | 56.35 / 48.39 | — | — |

| P3 온라인−오프라인 | AUROC 차이 / 95% CI (pp) | AP 차이 / 95% CI (pp) |
|---|---:|---:|
| DINOv3-L | +2.22 (+0.13..+4.33) | +0.82 (-0.68..+2.43) |
| V-JEPA 2.1-L | +0.76 (-1.64..+3.26) | +1.25 (-0.08..+2.87) |

온라인 P3의 V-JEPA−DINOv3 차이는 AUROC **-10.02 (-14.49..-5.18)pp**, AP **-7.52 (-12.08..-2.00)pp**다. CI는 장비별 영상 단위 paired bootstrap 1,000회로 계산했다. 정상 학습 경계와 별도 head·메모리가 모드별로 다르므로, 전체 프로토콜의 비교다. 실제 실시간 처리 성능은 아직 측정하지 않았다.

![Macro4 온라인·오프라인 정확도와 paired 차이](docs/figures/stage02/macro4_mode_comparison.png)

P3의 개선은 장비별로 달랐다. R01 DINOv3 P3는 전역 메모리 P0보다 낮았고, R03 V-JEPA P3는 AUROC 50% 부근이었다. V-JEPA 온라인 P2 평균이 P3보다 높다. 테스트에 맞춰 점수 부호·선택 epoch·임계값을 바꾸지 않았다. P0–P3 전체 수치, 장비별 편차, ROC/PR, 위상·시계열 그래프와 해석은 [오프라인 Stage 02](docs/STAGE02.md), [온라인·모드 비교 Stage 03](docs/STAGE03.md)에 제공한다.

IPAD 공개 모델의 R01 seed 0은 50 epoch 학습·평가를 완료했다. B0 픽셀 AUROC/AP **80.42/58.74%**, B1 특징 잔차 **65.70/46.65%**는 한 seed의 결과다. seed 1도 학습·평가를 완료했고 B0 **81.61/60.11%**, B1 **76.30/64.06%**다. seed 2의 50 epoch 학습도 완료됐으며, 평가 결과는 아직 미공개다. 아직 3-seed 기준선 결과가 아니다. 공개 코드 수정·재현 차이·CI·알람 실패 사례는 [Stage 01](docs/STAGE01.md)에 기록한다.

시간 이력 5·15·31의 144개 고정 백본 조건도 같은 정상 보정·평가 구간에서 비교했다. 온라인 두 백본의 장기 이력 개선량은 AUROC/AP 모두 CI에 0을 포함했다. [상세 수치·그래프·재현 검증](docs/ABLATION_HISTORY.md)을 제공하며 기본 이력 5를 유지한다.

이웃 수 k=1·5·10의 **144개 고정 백본 조건**을 완료했다. k=5의 기존 48개 실험 점수·알람과 정상 임계값을 검증했다. DINOv3 오프라인 k=10의 Macro4 개선량은 AUROC/AP **+0.72/+0.43pp**였으나 모든 백본·모드에서 일관된 우위는 없었다. [전체 수치·paired CI·그래프·검증 근거](docs/ABLATION_NEIGHBOURS.md)를 제공하며 기본 k=5를 유지한다.

두 백본의 R01 오프라인 LoRA **seed 0**은 20 epoch 학습·특징/메모리 재구성·실제 평가를 완료했다. P3 AUROC/AP는 DINOv3 **79.81/70.83%**, V-JEPA **33.06/27.52%**다. 같은 seed의 고정 백본 대비 DINOv3 P3는 개선됐지만 전역 P0는 악화됐고, V-JEPA P3 차이의 CI는 0을 포함했다. [상세 비교·CI·그래프](docs/LORA_R01_SEED0.md)를 제공하며 나머지 seeds·장비·온라인 LoRA는 진행 중이다. 정상 학습 곡선과 선택 근거는 [Stage 04](docs/STAGE04.md)에 있다. 순차 이벤트 평가와 30 FPS FIFO 측정 코드는 [Stage 05](docs/STAGE05.md)에 있으며, 실제 GPU 측정과 점수·알람 동등성 검증은 남아 있다. `—`는 미측정이다.

## 재현 환경

```bash
uv venv --python python3.12 .venv
source .venv/bin/activate
export UV_CACHE_DIR="$PWD/artifacts/uv-cache"
export MPLCONFIGDIR="$PWD/artifacts/mpl-cache"
export CUDA_CACHE_PATH="$PWD/artifacts/cuda-cache"
mkdir -p artifacts/tmp
export TMPDIR="$PWD/artifacts/tmp"
uv pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[test,gpu]'
export IPAD_DATA_ROOT=/absolute/path/to/IPAD_dataset/IPAD_dataset
python scripts/bootstrap_upstreams.py
ipad-audit --data-root "$IPAD_DATA_ROOT"
ipad-plot-audit
python -m pytest -q
```

CUDA 학습·검증은 GPU 접근이 허용된 환경에서 실행한다. 이 작업 환경에서는 샌드박스 내부 드라이버 접근이 제한됐으며, 샌드박스 밖에서 RTX PRO 6000과 BF16 연산을 확인했다.

## 모델 준비

```bash
python scripts/download_models.py --model vjepa21-l
python scripts/download_models.py --model dinov3-l --user-mirror
```

가중치는 `artifacts/weights`에 저장하며 Git에서 제외한다. [DINOv3 공식 README](https://github.com/facebookresearch/dinov3#pretrained-models)는 모델 사용 승인 후 이메일의 URL 사용을 안내한다.
기본 공식 주소는 HTTP 403, [공식 Hugging Face 모델](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)은 수동 승인/인증이 필요했다.
사용자가 지정한 [PIA-SPACE-LAB 배포 경로](https://huggingface.co/PIA-SPACE-LAB/dinov3-vitl-pretrain-lvd1689m)에서 동일 파일을 취득했다.
로컬 SHA256 `8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035`가 미러의 전체 해시와 일치하며, 공식 파일명 해시 접두어 일치와 공식 코드의 엄격한 가중치 로딩도 확인했다. 해당 배포 경로는 공식 Meta 호스팅이 아닌 사용자 제공 미러로 기록한다.

```bash
python scripts/smoke_backbone.py --model vjepa21-l \
  --upstream third_party/vjepa2 \
  --weights artifacts/weights/vjepa2_1_vitl_dist_vitG_384.pt \
  --sequence "$IPAD_DATA_ROOT/R01/training/frames/01"
```

V-JEPA 2.1의 엄격한 가중치 로딩과 실제 정상 16프레임 GPU 연산을 통과했다. 패치 출력은 `1×576×1024`, 위상 입력은 `1×1024`, 백본 학습 파라미터는 0이다.
모델 smoke 결과는 형태·엄격한 가중치 로딩·실제 영상 연산 검증이다. cold forward 시간을 FPS 또는 이상탐지 성능으로 해석하지 않는다.
두 백본 모두 동일한 출력 형태와 백본 고정을 확인했다. 근거는 [환경 메타데이터](results/setup/environment.json), [V-JEPA GPU smoke](results/setup/vjepa21-l_smoke.json), [DINOv3 GPU smoke](results/setup/dinov3-l_smoke.json)에 있다.

CI는 핵심 검증 **86개**와 업로드 검사를 실행한다. 정상 분할·위상·메모리·LoRA·기준선·resume·업로드 제외, Macro4 비중·paired bootstrap, 온라인 EOF 알람·FIFO 입력·이벤트 지연과 GPU 동시 작업 거부를 확인한다. LoRA 런타임의 선택 adapter·joint head·재구성 메모리와 paired 입력 출처, 이웃 수별 고정 메모리 검색·인과적 알람·PCA layout 보존도 검증한다. 재생 제어는 CPU 모형으로 검증했으며 실제 GPU 처리량을 입증하지 않는다. 실제 학습·추론 근거는 각 Stage에 별도로 기록한다.

전체 실험 행렬은 [experiment_matrix.yaml](configs/experiment_matrix.yaml)에 고정했다. 이 목록은 전체 요구 범위이며 완료 여부는 실제 결과로 확인한다.

## 실험·업로드 원칙

- 정상 학습 데이터만으로 PCA·메모리·LoRA 구성; 정상 검증으로 임계값 설정.
- 각 장비·seed·백본·모드의 공통 대상 프레임에서 비교; 테스트 영상별 min-max 정규화 금지.
- 온라인 입력은 `[t−15,…,t]`, 오프라인은 `[t−8,…,t+7]`; 실제 알람 지연에는 미래 프레임 대기도 포함.
- 단계별 코드·설정·지표 JSON/CSV·그래프 생성 코드·PNG/SVG를 함께 업로드.
- 원본 영상/이미지, 모델, 체크포인트, 특징 캐시, PCA·메모리 텐서는 업로드하지 않음.
- 업로드 전 `ipad-upload-check`; 파일별 10MiB 이하; 완료된 단계만 완료 태그 생성.

## 구현 상태

| 단계 | 상태 |
|---|---|
| 00 데이터 검증 | 감사·분할·그래프 완료, 라벨 정렬 검토 중 |
| 01 IPAD 기준선 | R01 seed 0의 50 epoch 학습·B0/B1 평가·CI/그래프 완료, seed 1 학습·평가 완료, seed 2의 평가 결과 미공개; 다른 장비·reference 조건 남음 |
| 02 오프라인 비교 | R01–R04 두 백본 3 seeds 완료, 두 백본 Macro4 확보 |
| 03 온라인 비교 | R01–R04 두 백본 3 seeds 완료, 온라인 Macro4 확보; 실시간 측정 남음 |
| 04 LoRA·추가 실험 | 두 백본 R01 오프라인 seed 0 정상 학습·실제 P0–P3 평가·paired CI/그래프 공개, 전체 3-seed 평가 진행 중 |
| 05 실시간·진단·재현성 | 순차 알람·FIFO·점수 일치 검증 구현, 이력 5/15/31 및 이웃 1/5/10 각각 144개 조건·paired CI/그래프 완료; 실제 GPU 측정·진단·나머지 제거 실험 남음 |

## 출처

[IPAD 논문](https://arxiv.org/abs/2404.15033) · [IPAD 코드](https://github.com/LJF1113/IPAD) · [DINOv3](https://github.com/facebookresearch/dinov3) · [V-JEPA 2.1](https://github.com/facebookresearch/vjepa2)

공식 upstream은 [고정 revision](configs/upstreams.json)으로 로컬에 취득한다. 모델/소스의 원 라이선스와 사용 조건을 따른다.
