# Stage 05 — 실시간 탐지와 진단

고정 백본의 이력 5·15·31 OFAT 144개 seed 조건을 완료했다. [공통 구간의 수치·paired CI·그래프와 재현 근거](ABLATION_HISTORY.md)를 제공한다. DINOv3 온라인 R01 seed 0의 첫 실제 FIFO 측정은 아래에 별도 기록하며 전체 모델·모드·장비·seed의 실시간 비교는 아직 완료되지 않았다. 특징 캐시 생성 시간, 모델 smoke 시간, LoRA 예비 학습 시간은 처리량 결과로 사용하지 않는다.

이웃 수 k=1·5·10의 두 백본 × 두 모드 × 네 장비 × 세 seeds, **144개 조건**도 완료했다. 기존 k=5의 48개 원본 조건을 점수·알람 차이 0으로 재현하고 정상 임계값 144개를 재계산했다. [Macro4·장비별 수치·paired CI·그래프·검색 출처](ABLATION_NEIGHBOURS.md)를 제공한다. 백본·모드 전체에 일관된 우위는 없어 기본 k=5를 유지한다.

메모리 크기와 bin 수의 **192개 GPU 조건·독립 검증을 완료**했다. 기본 48조건의 실제 배열·정상 temperature·점수·알람을 정확히 재현하고 정상 q99 192개를 재계산했다. V-JEPA의 4,096개 조건은 오프라인 AUROC/AP +4.79/+3.49pp, 온라인 +3.26/+2.94pp였고 두 모드의 두 paired CI 모두 0을 포함하지 않았다. [전체 Macro4·장비별 수치·paired CI·6개 그래프·검증 근거](ABLATION_MEMORY.md)를 제공하며 사전 기본 16 bins·2,048개를 유지한다.

8·16프레임 비교의 신규 48조건 중 DINOv3 오프라인 R01 세 seeds의 실제 평가·독립 검증을 완료했다. 8프레임 P3 AUROC/AP는 44.71/32.20%로, 16프레임 대비 −2.31/−1.01pp이며 두 paired CI는 0을 포함한다. [부분 결과·CI·그래프·입력 구간·재현 명령](ABLATION_CLIP_FRAMES.md)을 제공한다. 전체 Macro4와 실제 FPS/지연은 아직 없다.

## 준비된 스트리밍 점수와 평가

[runtime_state.py](../src/ipad_jepa/runtime_state.py)는 고정 정상 calibration과 주기 길이로 P3 점수와 3프레임 연속 초과 알람을 순차 계산한다. 최근 5개 위상만 유지하며 영상의 최종 길이·테스트 라벨을 받지 않는다. 온라인은 target 15부터 위상을 쌓고, target 19부터 알람이 유효하다. 오프라인은 target 8부터 위상을 쌓되 같은 target 19부터 알람이 유효하다.

온라인은 실제 마지막 7프레임도 계속 탐지한다. 정확도 비교의 `t=19..N−8` 마스크는 실행 후 보고에만 적용한다. 미래 EOF나 미확정 GT가 알람을 중단시키지 않도록 했다.

실행 후 GT를 읽어 이벤트별 최초 알람과 도착 시점 대비 지연을 계산한다. 지연에는 전처리·모델·검색·점수 처리, 입력 대기열, 오프라인의 7프레임 lookahead 대기가 포함돼야 한다. 평가 대상 프레임이 전혀 없는 GT 이벤트는 coverage 밖으로 기록한다. GT 경계가 미확정인 이벤트에는 이를 표시하고, 불확실한 onset의 지연은 제외한다. 정상으로 확인된 target에서 시작한 알람 구간 수와 정상 프레임의 알람 수를 함께 보고한다.

GT 이상 구간 안의 target에서 발생했더라도 실제 알람 출력은 이상 구간이 끝난 뒤 도착할 수 있다. 종료 경계가 확인된 이벤트에서는 첫 정상 프레임의 도착 이전에 출력된 알람, 그 이후의 늦은 알람, 종료 전 알람이 없는 이벤트를 별도로 집계한다. 종료 경계가 미확정인 이벤트는 이 집계에서 제외한다. 이는 관측된 이벤트 종료 시점을 기준으로 한 통계이며, 별도의 실시간 서비스 SLA를 가정하지 않는다.

테스트는 순차 시간 점수와 배치 계산의 일치, prefix 유지, 마지막 7프레임의 온라인 알람, 3프레임 warmup, 누락·중복 target 거부, 처리와 lookahead를 포함한 이벤트 지연, GT 미확정과 coverage 제외를 확인한다. 이는 실제 GPU 처리량 또는 알람 성능 측정이 아니다.

## 남은 실제 실행

30 FPS 도착, FIFO, 프레임 drop 없음, 메모리 갱신 없음으로 두 백본·두 모드를 비교한다. GPU에서 다른 학습·추론이 겹치지 않는 측정 조건을 기록한다. decode/resize, encoder, phase head, PCA/메모리 검색, score/alarm, queue/lookahead 대기를 분리하고 지속 FPS·p50/p95 지연·VRAM·queue 증가·miss·false alarm을 공개한다.

[benchmark_runtime.py](../scripts/benchmark_runtime.py)는 이 측정을 위한 실행 코드를 제공한다. 첫 완료 조건의 전체 영상 trace를 검증했으며 전체 측정과 최적화 동등성 검증은 남아 있다. 기본은 장비의 모든 실제 테스트 영상을 30 FPS 도착 시점에 맞춰 파일로 재생한다. `full`은 매 clip을 CPU 프레임 버퍼에서 조합·GPU로 전송하고 인코더를 재계산하며, `buffer`는 과거 픽셀 프레임을 GPU에 보관해 전송을 줄이고 같은 clip의 인코더를 재계산한다. DINO 온라인 FP32의 `reuse`는 과거 인코더 특징을 재사용하는 별도 조건이다. 정상 데이터로 20회 warmup 후 영상별 점수 상태를 초기화한다.

메모리 복원 시 저장된 PCA의 Fortran layout을 유지한다. 이웃 수 재현에서 기본 C layout 복사로 FP32 검색 점수가 달라졌음을 확인했으며, layout 보존 후 k=5가 기존 점수와 정확히 일치했다. 런타임의 실제 로더도 동일하게 수정하고 CPU stride 회귀 테스트를 추가했다. 실제 GPU 런타임의 점수·알람 동등성 게이트는 여전히 별도로 수행해야 한다.

입력 도착부터 실제 점수 계산 종료까지 벽시계 시간을 기록하고, GPU에 다른 compute PID가 있으면 측정을 거부한다. 검사 사이의 순간적인 외부 작업까지 감시하지는 않으므로 다른 실험 실행이 모두 끝난 상태에서 측정해야 한다. 원본 해시 검증은 측정 전 수행하므로 파일 캐시는 warm 조건이다. BF16 batch 1 점수와 기존 batch 4 캐시 점수의 차이, full/buffer 및 FP32 full/reuse의 점수·알람 일치도도 결과와 함께 검증해야 한다.

```bash
python scripts/benchmark_runtime.py --model dinov3-l --mode online \
  --device R01 --seed 0 --implementation full --precision bf16 \
  --data-root "$IPAD_DATA_ROOT" --upstream third_party/dinov3 \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --run artifacts/runs/dinov3-l/online/R01/seed0 \
  --results results/stage02/dinov3-l/online/R01/seed0 \
  --out results/stage05/runtime/dinov3-l/online/R01/seed0/full_bf16
```

GPU 측정이 완료된 경우에만 `runtime.json`을 완료 마커로 저장한다. 일부 영상을 지정한 실행은 전체 장비 평가로 표시하지 않는다. 기본 `--adaptation frozen`은 고정 백본과 해당 정상 메모리를 사용한다.

`--adaptation lora --lora-run <완료된 joint 학습 경로>`는 같은 seed의 정상 검증 CE로 선택된 adapter와 joint 위상 head를 사용한다. 20 epoch 전체 곡선·선택 epoch·학습/adapter 소스·기본 가중치를 검증하고, 실제 head 텐서가 선택 체크포인트와 같은지 확인한다. adapter 해시가 해당 seed에서 재구성한 정상 특징·PCA/메모리와 일치해야 하므로 고정 백본의 메모리를 대신 사용할 수 없다. 완료 전 학습, 다른 seed·teacher·head 또는 선택 epoch를 거부한다. 가중치와 adapter는 로컬에서만 읽는다.

```bash
# 해당 LoRA 학습·특징 재구성·정상 메모리 평가가 완료된 후 실행
python scripts/benchmark_runtime.py --model dinov3-l --mode online \
  --device R01 --seed 0 --adaptation lora \
  --lora-run artifacts/lora/dinov3-l/online/R01/seed0 \
  --implementation full --precision bf16 \
  --data-root "$IPAD_DATA_ROOT" --upstream third_party/dinov3 \
  --weights artifacts/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --run artifacts/runs_lora/dinov3-l/online/R01/seed0 \
  --results results/stage04/dinov3-l/online/R01/seed0 \
  --out results/stage05/runtime_lora/dinov3-l/online/R01/seed0/full_bf16
```

LoRA의 `full`/`buffer`, DINO 온라인 FP32 `full`/`reuse`도 각각 같은 선택 adapter·head·재구성 메모리로 비교해야 한다. 고정 백본의 특징 재사용 검증을 LoRA에 적용하지 않으며, 실제 LoRA 점수·알람 일치 게이트는 아직 실행하지 않았다. 체크포인트 선택과 혼합 조건 거부는 CPU 테스트 대상이며 실제 GPU 처리량을 뜻하지 않는다.

[verify_runtime_parity.py](../scripts/verify_runtime_parity.py)는 완료된 `full` 실행과 같은 precision의 `buffer` 또는 허용된 FP32 `reuse` 실행을 비교한다. 모델·위상 head·메모리·선택 adapter·teacher 손실 비중·입력 manifest·영상/라벨 해시가 같은지 확인하고, 알람이 유효한 모든 target의 위상·특징/시간 점수·최종 점수·알람을 비교한다. 온라인 마지막 7프레임과 미확정 GT도 제외하지 않는다. 사전에 고정한 게이트는 위상 circular 최대 차이 1e−5, raw 점수 `rtol=1e−5/atol=1e−7`, 최종 점수 `rtol=1e−5/atol=1e−4`, 알람 불일치 0개다. 게이트 통과는 해당 실제 측정 쌍에만 적용하며, 기존 batch 4 정확도 캐시와의 일치까지 의미하지 않는다. R01 고정 DINOv3 온라인 seed 0의 실제 BF16 full/buffer 게이트는 통과했고, FP32 full/reuse는 일부 점수 게이트에 실패했다. 아래 실제 비교 결과에 전체 영상의 근거를 기록한다.

```bash
python scripts/verify_runtime_parity.py \
  --reference results/stage05/runtime/dinov3-l/online/R01/seed0/full_bf16 \
  --candidate results/stage05/runtime/dinov3-l/online/R01/seed0/buffer_bf16 \
  --out results/stage05/runtime/dinov3-l/online/R01/seed0/full_buffer_parity.json
```

DINO full-clip BF16과 과거 특징 재사용 BF16의 수치 일치 게이트는 실패했다. FP32 특징 비교는 통과했지만, 실제 점수·알람과 실행 시간을 확인해야 한다. 재사용 구현의 숫자를 주 BF16 조건의 성능으로 대체하지 않으며, precision과 대응하는 full 계산 조건을 명시한다. 세부 근거는 [Stage 03](STAGE03.md)에 있다.

정상 진단 **17개 영상 × 49시나리오**의 stall/reverse·가림/국소 색 변화·36개 혼합 조합에 대해 전체 GPU 평가를 시작했다. [고정 위치/강도·네 증거 유형 정답·입력 문맥 도식·코드·검증 범위](DIAGNOSTICS.md)를 제공한다. 실제 confusion/Macro-F1은 전체 trace의 독립 검증 후 공개한다. 나머지 OFAT·LoRA·실시간 측정도 남아 있어 Stage 05 완료 태그는 생성하지 않는다.

## 작은 온라인 백본

DINOv3-B와 V-JEPA 2.1-B의 별도 768차원 정상 학습·평가 **24개 조건을 실행 중**이다. DINOv3 R01 온라인의 세 seeds는 L paired 기준과 독립 검증했다. P3 AUROC/AP는 B **43.86/31.35%**, L **53.69/36.92%**이며 B−L 두 CI 모두 0보다 낮다. [부분 정확도·CI·그래프·검증 한계·모델 출처·재현 경로](SMALL_BACKBONES.md)를 제공한다. 이 결론은 R01 온라인에 한정하며 전체 Macro4와 독립 실시간 성능은 아직 없다.

## 전체 실시간 측정 행렬과 재개

### 첫 실제 FIFO 측정

DINOv3-L 온라인 R01 seed 0, 고정 백본·P3·BF16·full 재계산의 **15개 전체 실제 테스트 영상**을 RTX PRO 6000에서 재생했다. 실제 자식 프로세스 종료 코드 0, 입력 3,685프레임·알람 유효 target 3,400개·공통 GT 지표 target 3,295개를 확인했다. 영상마다 정상 warmup 후 점수/큐 상태를 초기화하고, 30 FPS 도착·FIFO·drop 없음·고정 정상 메모리를 적용했다.

| 지표 | 실제 측정 |
|---|---:|
| 지속 입력 처리 FPS | 16.4630 |
| target 전체 지연 p50 / p95 | 3,471.09 / 7,172.03 ms |
| queue wait p95 | 6,895.15 ms |
| 영상별 최종 queue 증가 최대 | 10,948.55 ms |
| GPU peak allocated / reserved | 1,577.67 / 1,804.00 MiB |
| 평균 decode / encoder 포함 전송 | 2.36 / 58.17 ms |
| 평균 phase head / PCA·검색 / score·alarm | 0.50 / 0.99 / 0.06 ms |
| 평균 전체 service / feature cast | 62.28 / 0.18 ms |
| 공통 target AUROC / AP | 53.87 / 36.03% |
| operational GT 이벤트 탐지 / 미탐 | 1 / 7 (총 8개) |
| 이벤트 종료 전 알람 / 늦은 최초 알람 | 1 / 0 |
| 정상 알람 episode 시작 / 정상 알람 프레임 | 4 / 13 (평가 정상 2,169프레임) |

30 FPS 입력에 처리 속도가 미달해 대기열이 증가했다. p95는 전체 알람 유효 target의 벽시계 지연이며 영상별 p95의 평균이 아니다. 지속 FPS는 완료된 입력 간격 수를 첫 도착부터 마지막 처리 종료까지의 시간으로 나눈 값으로, 30 FPS 제한이 없는 최대 처리량 측정이 아니다. 관측된 단일 탐지 이벤트의 onset 지연은 1.3992초이며 한 이벤트로 지연 분포를 일반화하지 않는다.

![첫 실제 FIFO 비용·지연·대기열 측정](figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_bf16_full.png)

[원본 15개 CSV·runtime JSON·trace 검증](../results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16/full), [PNG/SVG와 그림 소스 해시](figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_bf16_full.json)를 제공한다. 실제 CSV의 도착 순서·GT·정상 보정·위상 이력·점수·알람·지표·이벤트와 전체 FPS/p50/p95/queue/cost를 재계산했고 PNG를 열어 범례·축·단위·출력 범위를 확인했다.

기존 batch 4 encoder 특징 캐시의 같은 seed P3 AUROC/AP는 53.94/36.11%로, runtime batch 1의 53.87/36.03%와 다르다. head는 기존 캐시 평가에서 여러 target을 batch로 처리한다. 동일 정상 보정과 공통 GT를 사용했지만 수치·알람 동등성을 선언하지 않는다. 첫 결과와 같은 조건의 full/buffer·FP32 full/reuse 비교를 아래에 추가했다. 여전히 한 seed·한 장비의 실제 파일 재생이며 반복 측정 CI, 전체 장비·LoRA 실시간 비교, 실제 카메라 입력은 아직 완료되지 않았다. GPU 격리는 setup과 각 영상 전후 다른 compute PID가 없는지 확인한 범위이며 검사 사이의 순간적인 외부 작업은 독립 감시하지 않았다.

```bash
PYTHONPATH=src:scripts python scripts/audit_runtime.py \
  --runtime results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16/full \
  --out results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16/full/trace_audit.json
PYTHONPATH=src:scripts python scripts/plot_runtime.py \
  --runtime results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16/full \
  --audit results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16/full/trace_audit.json \
  --out docs/figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_bf16_full
```

[run_runtime_matrix.py](../scripts/run_runtime_matrix.py)는 고정 백본·teacher=1 LoRA 각각의 두 모델 × 두 모드 × 네 장비 × 세 seeds, **96개 주 조건**을 유지한다. 각 조건의 BF16 full/buffer 192회와 DINOv3 온라인의 별도 FP32 full/reuse 48회, 합계 **240회**를 계획한다. 모든 실행은 해당 장비의 전체 실제 테스트 영상을 사용한다. 이 횟수는 완료 수가 아니다.

완료된 정확도·선택 head·정상 메모리가 있는 조건만 실행한다. 최초 준비 상태는 고정 백본 48조건과 R01 오프라인 LoRA 4조건, **52조건·128회**이며 나머지 LoRA 44조건은 대기로 남긴다. 각 실행 전후 소스·선택 head·메모리·정상 보정 해시를 확인하고, 자식 프로세스의 실제 종료 코드 0과 전체 영상 결과를 확인한 뒤 완료로 기록한다. 같은 조건의 full/buffer와 FP32 full/reuse 점수·알람 게이트가 수치적으로 실패하면 그 실패를 그대로 기록한다.

```bash
PYTHONPATH=src:scripts python scripts/run_runtime_matrix.py \
  --data-root "$IPAD_DATA_ROOT"
# 이전 controller와 측정 프로세스가 실제 종료된 후, 같은 소스로 재개:
PYTHONPATH=src:scripts python scripts/run_runtime_matrix.py \
  --data-root "$IPAD_DATA_ROOT" --resume
```

대형 GPU 런타임은 `.venv/bin/python`, 점수·알람 게이트는 별도 CPU 런타임을 사용한다. ledger는 `artifacts/tmp/runtime_matrix.json`에 저장한다. 커널 파일 잠금으로 controller 중복을 막고, 재개 시 boot ID·PID 시작 시점으로 이전 작업이 실제 종료됐는지 확인한다. 완료 기록의 CSV/JSON 해시를 확인하며 중단된 출력은 로컬 `artifacts/runtime_interrupted`에 보존한 뒤 해당 측정만 다시 실행한다. `artifacts/tmp/runtime_matrix.hold`가 있으면 현재 측정을 마친 뒤 다음 측정 시작 전에 대기하므로 독립 검증·업로드 작업을 GPU 측정과 겹치지 않게 수행할 수 있다.

추론용 선택 LoRA를 로딩한 뒤 전체 encoder 파라미터를 고정한다. eval만으로는 새 LoRA 파라미터의 gradient 플래그가 해제되지 않아 과거 특징 재사용 검사가 거부되는 문제를 수정했다. 선택된 nonzero adapter 텐서와 출력 보존·dropout 비활성화·인과적 ring 진입을 CPU 회귀 테스트로 확인한다. 실제 이벤트의 NumPy 불리언도 값과 null 경계를 유지하며 JSON으로 저장한다. 전체 영상 CSV를 저장하고 JSON 직렬화에서 실패한 최초 실행은 로컬에 보존하며 완료된 실시간 성능으로 집계하지 않는다.

[audit_runtime.py](../scripts/audit_runtime.py)는 완료된 측정 CSV에서 입력 순서·전체 프레임 수·도착/서비스/출력 시점·queue/lookahead/target latency를 재계산한다. 고정 정상 P3 median/MAD·q99, 실제 annotation·공통 mask, 위상 이력·최종 점수·3프레임 연속 알람, AUROC/AP와 이벤트 집계를 검증한다. 온라인 EOF와 미확정 GT도 실제 알람 검증에서 제외하지 않는다. 이벤트 정의는 기존 공용 함수를 재생하며, encoder/PCA/search 재실행·독립 VRAM 측정·반복 측정 CI를 주장하지 않는다.

[plot_runtime.py](../scripts/plot_runtime.py)는 이 검증을 다시 실행하고 같은 실제 CSV에서 영상별 FPS·latency p50/p95·처리 비용·사전 고정 영상 03의 FIFO 궤적을 PNG/SVG로 생성한다. p50/p95는 측정 target의 백분위이며 신뢰구간이 아니다. 한 seed의 파일 재생 수치를 최대 처리량·실제 카메라 결과·세-seed 평균으로 해석하지 않는다.


### 같은 조건의 버퍼·특징 재사용 실측

위와 같은 고정 DINOv3-L 온라인 R01 seed 0의 선택 head·정상 메모리·보정과 실제 테스트 15개 영상을 유지했다. 네 실행 모두 실제 종료 코드 0과 입력 3,685개·알람 유효 target 3,400개·공통 GT target 3,295개를 확인했다. 각 저장 trace를 독립 재생해 점수·알람·queue·시간·지표를 검증했다. BF16/FP32는 별도 수치 조건이며 구현 비교는 같은 precision의 full을 기준으로 한다.

| precision / 구현 | 지속 입력 FPS | target 지연 p95 (ms) | peak allocated (MiB) | 공통 AUROC / AP (%) |
|---|---:|---:|---:|---:|
| BF16 / full | 16.4630 | 7,172.03 | 1,577.67 | 53.8665 / 36.0274 |
| BF16 / buffer | 17.0509 | 6,313.83 | 1,605.99 | 53.8665 / 36.0274 |
| FP32 / full | 5.0911 | 44,064.40 | 1,634.58 | 53.6205 / 35.7571 |
| FP32 / reuse | 29.9135 | 37.64 | 1,317.05 | 53.6201 / 35.7569 |

BF16 full/buffer는 15개 영상의 위상·특징/시간 점수·최종 점수·알람 차이가 모두 0이었다. GPU 픽셀 버퍼는 전송/clip 조합을 줄이지만 encoder를 매번 재계산하므로 이 조건에서 30 FPS 입력의 queue 증가를 해소하지 못했다.

**FP32 full/reuse의 사전 점수 게이트는 실패했다.** 위상과 시간 점수 게이트는 통과했고 실제 알람 불일치는 0개였으나, 영상 03·11·13에서 특징 잔차와 최종 점수가 허용 오차를 초과했다. 아래 수치는 영상의 모든 알람 유효 target 중 최대 절대 차이이며, 고정 게이트는 위 절의 `allclose(rtol, atol)` 그대로 유지한다.

| 영상 | 특징 잔차 최대 절대 차이 | 최종 점수 최대 절대 차이 | 알람 불일치 |
|---|---:|---:|---:|
| 03 | 0.0002212226 | 0.0010887255 | 0 |
| 11 | 0.0004786253 | 0.0023368037 | 0 |
| 13 | 0.0004116595 | 0.0020055073 | 0 |

초기 네 clip의 FP32 특징 오차 통과는 전체 영상의 downstream 점수 동등성을 입증하지 못했다. 처리 속도와 알람 일치만으로 점수 동등성을 선언하거나 허용 오차를 늘리지 않는다. 수치 차이의 원인과 다른 조건의 재현 여부는 추가 확인 대상이다. BF16 주 조건에 FP32 reuse 속도를 대입하지 않으며, 29.91 FPS는 30 FPS 도착으로 제한한 파일 재생 결과여서 최대 처리량을 뜻하지 않는다.

![precision별 실시간 처리량·지연과 실패한 점수 게이트](figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_pairs.png)

지연 패널은 로그 축이며 숫자를 ms로 함께 표시했다. p95는 실제 target 분포의 백분위이고 CI가 아니다. [그림 PNG/SVG·수치·소스 해시](figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_pairs.json), [BF16 원본·게이트](../results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/bf16), [FP32 원본·게이트](../results/stage05/runtime/frozen/dinov3-l/online/R01/seed0/fp32)를 제공한다. trace 재생 통과와 full/candidate 점수 동등성 게이트를 구분한다. 전체 240회 중 이 한 주 조건의 네 실행이 완료됐으며 세-seed 실시간 평균·LoRA/V-JEPA의 실시간 우열·반복 측정 불확실성은 아직 없다.

```bash
PYTHONPATH=src:scripts python scripts/plot_runtime_pairs.py \
  --root results/stage05/runtime/frozen/dinov3-l/online/R01/seed0 \
  --data-root "$IPAD_DATA_ROOT" \
  --out docs/figures/stage05/runtime/dinov3-l_online_R01_seed0_frozen_pairs
```

[plot_runtime_pairs.py](../scripts/plot_runtime_pairs.py)는 네 trace 감사와 두 저장 gate를 실제 소스로 다시 계산한다. 실패 기록을 통과로 바꾼 파일이나 precision 간 선택 head/메모리/데이터가 다른 조건을 거부한다. 비교 그림은 실제 측정치와 실패 상태를 함께 보존하며 세-seed 평균·CI를 만들지 않는다.
