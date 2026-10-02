# 작업 흐름 정리 (최종 README 작성용 메모, jjyu1018)

> 용도: 최종 `README.md`를 쓸 때 근거로 삼는 작업 기록. 숫자는 모두 `notebooks/02_DAY2_Modeling.ipynb` 또는 아래 "별도 분석" 출력에서 확인한 값.
> 마지막 갱신: 2026-10-02

## 1. 전체 흐름

```
EDA (01_DAY1_EDA / 01_EDA, 가설 H1·H2)
  └─ 방향 확정: cycle_life 회귀, deltaQ_logvar 1개로 시작, 충전 정책·온도·초기 용량은 feature에서 제외
       │
       ├─[내 쪽] 02_MODEL_V2.ipynb  (별도 구현, 최종 저장소에는 하나만 남김)
       │     · 6개 모델 × 입력 A/B, 정책 단위 그룹 CV(5회 반복) + 중첩 튜닝, hold-out 정책 5종
       │     · 가설 H3(log 타깃)·H4(외삽)·H5(장수명 과소예측), 앙상블 시도
       │     · Batch 2 평가 1회 → 31.5%, 배치 간 차이 확인, 논문 비교용 분류(부록)
       │
       └─[팀원] 02_DAY2_Modeling.ipynb  (최종 저장소에 남기는 파이프라인)
             · manifest로 분할 고정(split_manifest_v1.csv, SHA-256 0534c12c…9652)
             · Feature Set A/B/C 사전 정의, 후보 모델·하이퍼파라미터 사전 고정
             · Hold-out 확인 전 후보 잠금 → Batch 2 단일 평가 → 31.53%
             · results/modeling/ 에 설정·결과 JSON/CSV/그래프 저장, src/modeling_utils.py

두 구현의 결론이 같음: 모델·feature 동일, Test(Batch 2) 31.5% vs 31.53%
  └─ 하나만 남기기로 결정 → 팀원 노트북을 기준으로, 내 쪽 고유 분석은 노트북 13번 섹션 + 이 문서로 이관
```

## 2. 최종 결과 (팀 노트북 기준, 회귀 Reporting format)

| 구분 | MAPE (%) | 비고 |
|---|---|---|
| Train (Batch 1 CV) | 7.60 ± 1.57 | `GroupKFold(5)`, 35셀 |
| Valid (Batch 1 Hold-out) | 11.63 | 정책 5종 11셀(manifest 고정) |
| Test (Batch 2) | 31.53 | 39셀, 1회 평가. MAE 155.59, RMSE 169.56, R² 0.402 |
| Gap (Train-Valid) | +4.03%p | (+) 과적합 의심 → 단정하지 않음(아래 3번 참고) |
| Gap (Valid-Test) | +19.90%p | (+) 배치 간 일반화 저하 의심 → 지지 |
| Gap (Target-Test) | +22.43%p | Target: 원논문 9.1% |

- 최종 모델: `deltaQ_logvar` 1개 + Linear Regression. 모델 선택은 Batch 1 내부 검증(CV 평균·표준편차)과 설명 가능성·외삽 특성으로 함. Batch 2 점수를 보고 바꾸지 않음.

## 3. 핵심 해석과 근거

1. **모델 복잡도가 아니라 배치 간 일반화 문제.** Batch 1 안은 7~12%인데 Batch 2에서만 31.5%. 앙상블·비선형(트리)으로도 해결되지 않음(별도 분석).
2. **학습 범위 문제(외삽).** Batch 2의 30/39셀(76.9%)이 Batch 1 최소 수명(534)보다 짧고, 이 30셀은 100% 과대예측. 단수명 셀 평균 APE 36.7%, 나머지 14.2%.
3. **Gap(Train-Valid) +4.03%p는 과적합으로 단정하지 않음.** Valid 11셀 중 한 정책(`3_6C-80PER_3_6C`, 수명 약 1,180, 평균 APE 19.2%)이 평균을 끌어올림. 이 정책을 빼면 약 8.8%(노트북 13.3 ②, 정책을 제외한 재학습은 안 함 = 가설).
4. **배치 간 차이는 확인했지만 원인은 미확정**(노트북 13.1):
   - 단수명 30셀의 온도·IR은 대부분 Batch 1 범위 안, 정책 9종 중 Batch 1에도 있는 것은 2종뿐.
   - 같은 `deltaQ_logvar` 구간(−3.71~−3.09)에서 실제 수명이 Batch 1 14셀 평균 659 vs Batch 2 단수명 453(약 206 사이클 짧음) = 절편 이동.
   - 가진 값 중 이 차이를 설명하는 것은 없음. "Batch 1에 없는 충전 조건의 영향이 `deltaQ_logvar`에 담기지 않았을 가능성"은 후보일 뿐.

## 4. 논문 비교

- 논문: Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 383–391 (DOI 기준 2019). LFP/graphite 124셀, 수명 150~2,300사이클.
- Target(과제 안내문 기준): 회귀 테스트 오차 9.1%(초기 100사이클), 분류 테스트 오차 4.9%(초기 5사이클).
- **본 분석은 회귀 하나**(안내문: 회귀 / 분류 중 한 가지). 회귀 Gap(Target-Test) +22.4%p.
- **분류는 논문 비교용 참고**(노트북 13.2): 회귀 예측을 550 기준으로 이진화. Test 1−Acc 59.0%(Gap +54.1%p, 논문 4.9%). 입력이 달라(논문 초기 5사이클 직접 분류 vs 초기 100사이클 회귀의 이진화) 같은 조건의 비교가 아님. Valid는 11셀 전부 장수명이라 평가 불가. **Reporting format·모델 선택·결론에는 쓰지 않음.**
- **직접 확인하지 못한 것**: 논문 Table 1의 모델별 수치, 9.1%의 지표 정의와 테스트셋 구성. abstract 수준의 수치(100사이클 9.1%, 5사이클 분류 4.9%)와 과제 안내문의 값만 사용.

## 5. 두 구현 비교 (참고)

| | 팀 노트북(남김) | 내 노트북(별도 분석) |
|---|---|---|
| 분할 | manifest 고정, `GroupKFold(5)` | `GroupShuffleSplit(test_size=5, 42)`, 정책 단위 5-fold ×5회 |
| Hold-out 정책 | 5종(11셀) | 5종(11셀), **2종이 다름**(`5.4C(60%)-3C`/`6C(50%)-3C` vs 팀 `…-3.6C`) |
| 하이퍼파라미터 | 사전 고정, Grid Search 없음 | train fold 안 중첩 CV 튜닝 |
| Train CV / Valid / Test | 7.60% / 11.63% / 31.53% | 7.3% / 13.1% / 31.5% |

- Test는 분할 방식에 영향받지 않으므로 직접 비교 가능(같음). Train/Valid는 분할이 달라 직접 비교하지 않음.
- Valid 차이(11.63 vs 13.1)는 hold-out 정책 구성 차이로 보이나 같은 manifest로 재계산해 확인한 것은 아님(가설).
- 정책 이름 표기가 다름(`3_6C-80PER_3_6C` = `3.6C(80%)-3.6C`). 같은 `test_size`·`random_state=42`인데 hold-out이 달랐던 이유(정책 라벨·정렬 순서 차이로 추정)는 확인하지 못함.

## 6. 파일 지도

| 파일 | 역할 | 상태 |
|---|---|---|
| `notebooks/02_DAY2_Modeling.ipynb` | **최종 모델링 파이프라인**(팀원) + 13번 추가 분석(jjyu1018) | 남김 |
| `notebooks/02_MODEL_V2.ipynb` | 내 별도 구현. 13번 섹션에 고유 분석 이관 후 정리 대상 | 팀 합의 후 제거 |
| `results/modeling/` | 설정·결과·예측·그래프 (manifest 포함) | 팀원 생성 |
| `src/modeling_utils.py` | 파이프라인 헬퍼 | 팀원 생성 |
| `MODELING_SUMMARY_jjyu1018.md` | 내 쪽 시도·과정·결과·논문 비교(회귀 본 분석, 분류 부록) | 보조 문서 |
| `NOTES.md` | 일자별 버전 이력(v0.x)과 결정 사항 | 기록용 |
| `TEAM_MODEL_DEVELOPMENT_SUMMARY.md` | 팀원 정리 | 팀원 작성 |

## 7. 최종 README에 넣을 것 / 확인할 것

- [ ] 문제 정의: 초기 100사이클 → `cycle_life` 회귀(분류는 논문 비교용 참고)
- [ ] 데이터 역할: Batch 1 학습(Train 35 / Valid 11), Batch 2 테스트(39), Batch 3 EDA·추가 확인
- [ ] 누수 방지: 정책 단위 분할, manifest 고정, 전처리는 Pipeline 안에서 fold Train만 fit, Batch 2는 1회 평가
- [ ] 결과 표(2번)와 Gap 해석(3번), 논문 비교(4번, 직접 확인 못 한 항목 명시)
- [ ] 한계: 소표본(Batch 1 46셀, 정책 23종), Valid 11셀 불안정, 배치 간 차이 원인 미확정, 논문 Table 1 미확인
- [ ] 시도했으나 채택하지 않은 것: 앙상블, log 타깃, 비선형 트리(노트북 13.3 ③) — 별도 분석 수치라 재현 여부 표시
- [ ] AI 도구 활용 여부와 고민 과정은 성능이 아니라 평가 대상이라고 안내됨: 위 기록을 과정 서술에 활용
- [ ] **NOTES.md 섹션 6 잔여 항목**: README 문장 수정(수명 결측 설명, Q3 문구, 입력 feature·충전조건, 검증 분할 정책 단위), 캐시 추적 해제와 `.gitignore`(`data/`, `.venv/`)
- [ ] 확인 필요: `data/cache/`가 현재 작업 트리에서 삭제됨(내 노트북 재실행 불가, 팀 노트북은 `results/battery_eda/cell_features.csv`로 재현 가능)
- [ ] 확인 필요: 병합 중(`git status`) — 두 요약 md의 위치(`docs/`로 이동 여부)
- [ ] 보고서는 예쁘게 만들 필요 없음(안내문). 고민의 과정이 보이게
