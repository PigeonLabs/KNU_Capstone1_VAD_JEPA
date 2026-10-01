# Stage 00 — 데이터 감사와 정상 영상 분할

## 실제 확인 결과

R01–R04를 현재 로컬 데이터에서 다시 검사했다. 학습 111개 영상/50,642프레임, 테스트 66개 영상/33,462프레임이다.
모든 검사 영상의 JPG 번호는 0부터 연속이며, 테스트 영상과 라벨 파일의 ID 대응을 확인했다.
라벨은 1차원 0/1 배열이다. 이미지 디코딩 검사는 영상별 처음·중간·끝 3프레임에 한정하며, 전체 JPG의 무결성 검사를 완료했다고 해석하지 않는다.

최종 플랜의 정확한 분할 수를 적용했다. R01 23/5/6, R02 21/5/4, R03 15/4/3, R04 17/4/4이며 순서는 학습/정상 검증/진단이다.
단순 15% 올림은 R01에 6개 검증 영상을 배정하므로, 계획표의 장비별 검증 수를 명시적으로 사용한다.
기본 `numpy.random.default_rng(42)`를 장비별로 독립 초기화한다. 실제 영상 ID는 [splits.json](../results/stage00/splits.json)에 기록되어 있다.

## 해결되지 않은 라벨 정렬

| 장비/영상 | JPG | 라벨 | 라벨 전환 위치 |
|---|---:|---:|---|
| R02/12 | 806 | 805 | 287, 557 |
| R02/13 | 609 | 608 | 11, 28, 303, 317 |
| R02/14 | 497 | 498 | 134, 184, 340, 390 |

세 영상은 `review_required`이다. 길이를 맞추기 위해 임의로 끝을 자르거나 라벨을 복제하지 않았다.
정렬 검토가 완료되기 전 전체 66개 영상의 확정 성능을 발표하지 않는다. 영상 제외 없이 미확정 프레임 마스크와 정렬 근거를 작성할 예정이다.
`audit.json`의 `label_alignment_complete`는 현재 false이며 Stage 00 전체 완료 태그는 아직 생성하지 않는다.

## 재현

```bash
ipad-audit --data-root "$IPAD_DATA_ROOT" --out results/stage00
ipad-plot-audit
python -m pytest -q
```

그래프는 감사 JSON에서만 생성하며 원본 프레임을 포함하지 않는다. PNG/SVG를 직접 렌더링해 축·단위·표본 수·텍스트 잘림을 확인했다.
파일명 해시는 영상의 프레임 순서를 확인하는 용도이며, JPG 콘텐츠 전체 해시가 아니다.

## 출처와 검증 범위

- 원본: 로컬 `IPAD_dataset/IPAD_dataset/R01..R04`.
- 수치 근거: [audit.json](../results/stage00/audit.json).
- 영상 메타데이터·라벨 파일 해시: [manifest.json](../results/stage00/manifest.json).
- 분할 근거: [splits.json](../results/stage00/splits.json).
- 소스 프로토콜: [IPAD 원 논문](https://arxiv.org/abs/2404.15033), [공식 코드](https://github.com/LJF1113/IPAD).

