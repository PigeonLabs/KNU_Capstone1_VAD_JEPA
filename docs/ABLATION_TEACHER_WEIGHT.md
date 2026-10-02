# LoRA의 teacher 손실 비중 비교

기존 주 LoRA 조건의 `λ_teacher=1`과 새 `λ_teacher=0`을 비교한다. 두 L 백본 × 온라인/오프라인 × 네 장비 × 세 seeds의 **신규 48조건**에서 정상 데이터만 사용한다. 큰 모델·원본 영상·특징·체크포인트는 GitHub에 올리지 않는다.

가중치 0은 **teacher 항이 최적화 손실에 기여하지 않는 처리**다. 같은 학습 코드를 사용하므로 고정 teacher 캐시를 입력 검증에 읽고 dense L2도 모니터링 값으로 계산한다. teacher 읽기/연산 비용을 제거한 구현이나 학습 속도 비교로 해석하지 않는다.

```mermaid
flowchart LR
  A[같은 정상 fit / calibration] --> B[같은 초기 백본 · seed · q/v LoRA]
  B --> C[Phase CE + λ × 고정 teacher dense L2]
  C --> D[λ=0 또는 기존 λ=1]
  D --> E[20 epoch · 최소 정상 calibration CE 선택]
  E --> F[선택 adapter와 joint head로 특징 재추출]
  F --> G[새 PCA · 프로토타입 · 정상 보정]
  G --> H[같은 실제 test · 공통 GT 구간]
```

| 항목 | 두 처리에서 고정 |
|---|---|
| 데이터 | IPAD 실제 정상 fit 76 / calibration 18 영상, 전체 영상 단위 분할 seed 42 |
| 입력 | RGB 384×384, 16프레임; 모드별 기존 입력/target 규칙 |
| 학습 | 마지막 4개 블록 q/v LoRA, rank 8, alpha 16, dropout 0.05 |
| optimizer | encoder LR 1e−4, head LR 1e−3, micro batch 1, accumulation 8, BF16 |
| 선택 | 20 epoch를 모두 수행한 뒤 최소 정상 calibration phase CE |
| 메모리 | 선택 특징으로 PCA 256·16 bins/2,048 prototypes·온도·MAD·q99 재학습 |
| 평가 | 기존 P0–P3; 주 비교 P3; `t=19..N−8`와 공유 라벨 불확실성 제외 구간 |
| 통계 | 같은 장비/seed/원본 영상의 paired 비교, 3-seed 지표 평균, 영상 bootstrap 1,000회 |

`λ=1` 기준은 `results/stage04`의 **같은 20-epoch joint LoRA 조건**이다. 고정 백본 결과를 teacher 1로 대체하거나 서로 다른 seed/head/메모리를 쌍으로 만들지 않는다. 원본 teacher와 초기 가중치·코드는 두 처리에서 같고, 새 처리에서는 adapter·joint head·전체 특징과 메모리를 다시 구성한다.

## 실행과 완료 근거

[실행기](../scripts/run_teacher_weight_matrix.py)는 기존 캐시·학습·결과와 겹치지 않는 새 경로만 허용한다. train 명령에 `--teacher-weight 0.0`을 명시하고 완료된 metadata와 전체 곡선으로 실제 처리와 최소 정상 CE 선택을 확인한다. pilot, 부분 epoch, 다른 seed 또는 teacher 1 결과를 거부한다. 새 프로토콜 검증 테스트 8개를 통과했다. 이는 실제 GPU 학습·정확도의 완료 근거와 구분한다.

```bash
export PYTHONPATH=src
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export TMPDIR="$PWD/artifacts/tmp"
export CUDA_CACHE_PATH="$PWD/artifacts/cuda-cache"
python scripts/run_teacher_weight_matrix.py --data-root "$IPAD_DATA_ROOT"
```

기본 명령은 신규 48조건을 실행한다. 학습은 `artifacts/lora_teacher0`, 선택 특징은 `artifacts/features_lora_teacher0`, head/메모리는 `artifacts/runs_lora_teacher0`, 공개 가능한 조건 결과는 `results/stage05/ablations/teacher_weight/T0`에 저장한다. 현재 실행 상태는 로컬 `artifacts/tmp/teacher0_pipeline.json`에 기록한다. 새 경로를 요구하므로 실행 중인 행렬에 명령을 중복 호출하지 않는다.

코드·manifest·설정을 해시로 고정하고 각 subprocess 전후에 변경 여부를 확인한다. 각 완료 조건의 공개 CSV/JSON, 선택 adapter·head·메모리, 실제 캐시 메타데이터와 target 배열의 해시를 남긴다. 디스크에는 다음 조건의 전체 특징과 기본 64 GiB 여유를 확보한 뒤 새 학습을 시작한다. 완료된 derived 캐시의 보관·정리는 독립 검증과 재현 근거를 유지하며 처리해야 한다.

[실제 신규 48조건의 시작·소스 고정·8개 테스트 근거](../results/setup/teacher_weight_implementation_check.json)를 제공한다. 첫 조건은 DINOv3-L 오프라인 R01 seed 0이며 실제 학습 subprocess의 `--teacher-weight 0.0` 인자를 확인했다. 이 시작 기록은 optimizer update·epoch·전체 GPU 학습의 완료를 증명하지 않는다.

**현재 정확도·CI·전체 Macro4·실시간 결과는 없다.** 시작/완료 표시는 실제 ledger·조건 증명으로 구분한다. teacher 0/1 paired 집계·독립 점수/GT 검증과 그래프 공개는 해당 완료 조건이 확보된 뒤 수행한다. Stage 05 전체 완료 태그는 남은 OFAT·진단·실시간 비교까지 검증한 뒤에만 생성한다.
