# Stage 03 — 온라인 탐지와 추론 구현 비교

## 현재 실행 범위

두 백본의 고정 특징 추출을 온라인 `[t−15,…,t]` 창으로 시작했다. R01–R04의 정상 fit/calibration과 테스트를 순서대로 처리한다.
정상 fit 초기 구간만 첫 프레임 왼쪽 패딩을 허용하며, 검증·테스트는 실제 과거 16프레임이 확보된 뒤 시작한다. 정확도 비교의 공통 대상은 오프라인과 같은 `t=19..N−8`이다.

온라인 학습·평가가 완료되면 각 장비의 P0–P3와 오프라인 차이를 같은 영상 단위 bootstrap으로 비교한다. 현재 온라인 AUROC/AP 및 실제 FPS·알람 지연 결과는 아직 없다.

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
