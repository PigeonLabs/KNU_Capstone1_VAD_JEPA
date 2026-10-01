# Stage 05 — 실시간 탐지와 진단

고정 백본의 이력 5·15·31 OFAT 144개 seed 조건을 완료했다. [공통 구간의 수치·paired CI·그래프와 재현 근거](ABLATION_HISTORY.md)를 제공한다. 실제 FPS·지연·알람 지연은 아직 측정하지 않았다. 특징 캐시 생성 시간, 모델 smoke 시간, LoRA 예비 학습 시간은 처리량 결과로 사용하지 않는다.

## 준비된 스트리밍 점수와 평가

[runtime_state.py](../src/ipad_jepa/runtime_state.py)는 고정 정상 calibration과 주기 길이로 P3 점수와 3프레임 연속 초과 알람을 순차 계산한다. 최근 5개 위상만 유지하며 영상의 최종 길이·테스트 라벨을 받지 않는다. 온라인은 target 15부터 위상을 쌓고, target 19부터 알람이 유효하다. 오프라인은 target 8부터 위상을 쌓되 같은 target 19부터 알람이 유효하다.

온라인은 실제 마지막 7프레임도 계속 탐지한다. 정확도 비교의 `t=19..N−8` 마스크는 실행 후 보고에만 적용한다. 미래 EOF나 미확정 GT가 알람을 중단시키지 않도록 했다.

실행 후 GT를 읽어 이벤트별 최초 알람과 도착 시점 대비 지연을 계산한다. 지연에는 전처리·모델·검색·점수 처리, 입력 대기열, 오프라인의 7프레임 lookahead 대기가 포함돼야 한다. 평가 대상 프레임이 전혀 없는 GT 이벤트는 coverage 밖으로 기록한다. GT 경계가 미확정인 이벤트에는 이를 표시하고, 불확실한 onset의 지연은 제외한다. 정상으로 확인된 target에서 시작한 알람 구간 수와 정상 프레임의 알람 수를 함께 보고한다.

GT 이상 구간 안의 target에서 발생했더라도 실제 알람 출력은 이상 구간이 끝난 뒤 도착할 수 있다. 종료 경계가 확인된 이벤트에서는 첫 정상 프레임의 도착 이전에 출력된 알람, 그 이후의 늦은 알람, 종료 전 알람이 없는 이벤트를 별도로 집계한다. 종료 경계가 미확정인 이벤트는 이 집계에서 제외한다. 이는 관측된 이벤트 종료 시점을 기준으로 한 통계이며, 별도의 실시간 서비스 SLA를 가정하지 않는다.

테스트는 순차 시간 점수와 배치 계산의 일치, prefix 유지, 마지막 7프레임의 온라인 알람, 3프레임 warmup, 누락·중복 target 거부, 처리와 lookahead를 포함한 이벤트 지연, GT 미확정과 coverage 제외를 확인한다. 이는 실제 GPU 처리량 또는 알람 성능 측정이 아니다.

## 남은 실제 실행

30 FPS 도착, FIFO, 프레임 drop 없음, 메모리 갱신 없음으로 두 백본·두 모드를 비교한다. GPU에서 다른 학습·추론이 겹치지 않는 측정 조건을 기록한다. decode/resize, encoder, phase head, PCA/메모리 검색, score/alarm, queue/lookahead 대기를 분리하고 지속 FPS·p50/p95 지연·VRAM·queue 증가·miss·false alarm을 공개한다.

[benchmark_runtime.py](../scripts/benchmark_runtime.py)는 이 측정을 위한 실행 코드를 제공한다. 아직 실제 GPU 측정으로 검증하지 않았다. 기본은 장비의 모든 실제 테스트 영상을 30 FPS 도착 시점에 맞춰 파일로 재생한다. `full`은 매 clip을 CPU 프레임 버퍼에서 조합·GPU로 전송하고 인코더를 재계산하며, `buffer`는 과거 픽셀 프레임을 GPU에 보관해 전송을 줄이고 같은 clip의 인코더를 재계산한다. DINO 온라인 FP32의 `reuse`는 과거 인코더 특징을 재사용하는 별도 조건이다. 정상 데이터로 20회 warmup 후 영상별 점수 상태를 초기화한다.

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

[verify_runtime_parity.py](../scripts/verify_runtime_parity.py)는 완료된 `full` 실행과 같은 precision의 `buffer` 또는 허용된 FP32 `reuse` 실행을 비교한다. 모델·위상 head·메모리·선택 adapter·teacher 손실 비중·입력 manifest·영상/라벨 해시가 같은지 확인하고, 알람이 유효한 모든 target의 위상·특징/시간 점수·최종 점수·알람을 비교한다. 온라인 마지막 7프레임과 미확정 GT도 제외하지 않는다. 사전에 고정한 게이트는 위상 circular 최대 차이 1e−5, raw 점수 `rtol=1e−5/atol=1e−7`, 최종 점수 `rtol=1e−5/atol=1e−4`, 알람 불일치 0개다. 게이트 통과는 해당 실제 측정 쌍에만 적용하며, 기존 batch 4 정확도 캐시와의 일치까지 의미하지 않는다. 아직 실제 GPU 비교 결과는 없다.

```bash
python scripts/verify_runtime_parity.py \
  --reference results/stage05/runtime/dinov3-l/online/R01/seed0/full_bf16 \
  --candidate results/stage05/runtime/dinov3-l/online/R01/seed0/buffer_bf16 \
  --out results/stage05/runtime/dinov3-l/online/R01/seed0/full_buffer_parity.json
```

DINO full-clip BF16과 과거 특징 재사용 BF16의 수치 일치 게이트는 실패했다. FP32 특징 비교는 통과했지만, 실제 점수·알람과 실행 시간을 확인해야 한다. 재사용 구현의 숫자를 주 BF16 조건의 성능으로 대체하지 않으며, precision과 대응하는 full 계산 조건을 명시한다. 세부 근거는 [Stage 03](STAGE03.md)에 있다.

진단 17개 정상 주기의 stall/reverse, occlusion/local colour, 혼합 변환과 4종 evidence confusion matrix·macro-F1, clip/bin/budget/k/표현/teacher weight 제거 실험 및 작은 온라인 백본도 [고정 실험 행렬](../configs/experiment_matrix.yaml)에 따라 수행해야 한다. 완료된 실측 근거가 없으므로 Stage 05 완료 태그를 생성하지 않는다.

## 작은 온라인 백본 준비

V-JEPA 2.1 ViT-B의 가중치를 고정된 공식 README 경로에서 취득했다. [다운로드 출처·크기·SHA256](../results/setup/vjepa21-b.json)을 공개하며, 1.66 GB 모델 파일은 로컬에만 보관한다. `python scripts/download_models.py --model vjepa21-b`로 재현할 수 있다. 엄격한 모델 로드·768차원 특징 처리·이상탐지 평가·실시간 측정은 아직 수행하지 않았다. 이 준비 기록을 작은 백본 실험 결과로 해석하지 않는다.
