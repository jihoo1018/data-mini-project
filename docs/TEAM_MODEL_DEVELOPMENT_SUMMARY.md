# 팀원 공유용 모델 개발 정리

## 한눈에 보는 요약

저희 모델은 초기 100사이클의 `deltaQ_logvar`로 배터리 수명을 예측하는 Linear Regression입니다. 같은 충전 `policy`가 학습과 검증에 섞이는 것을 막기 위해 Batch 1을 policy 단위 Hold-out과 5-fold Group CV로 나눴고, 분할 결과는 manifest로 고정했습니다. 여섯 종류의 후보 모델과 세 가지 Feature Set을 Batch 1에서 비교한 결과 단일 Feature 선형회귀가 가장 낮고 안정적인 CV MAPE를 보여 최종 선택했습니다. Batch 1 CV는 7.60±1.57%, 고정 Valid는 11.63%였지만 Batch 2에서는 31.53%로 성능이 저하됐습니다. 이는 Batch 2의 76.9%가 학습 수명 범위보다 짧아 단수명 셀을 모두 과대예측한 외삽 문제와 배치 분포 이동 때문으로 해석했습니다. 따라서 이 모델은 정확한 교체 시점 결정이 아니라 초기 위험 셀 선별을 보조하는 모델로 한정했습니다.

---

## 1. 분석 목적과 과제 정의

초기 100사이클에서 계산할 수 있는 특성으로 배터리의 최종 수명인 `cycle_life`를 예측하는 **회귀 문제**로 정의했습니다. 단순히 가장 낮은 오차를 만드는 것보다 초기 열화 신호와 수명의 관계를 설명하고, 새로운 충전조건과 배치에 적용할 수 있는지를 확인하는 데 목적을 두었습니다.

모델을 선택할 때 다음을 함께 고려했습니다.

- Batch 2를 모델 선택에 사용하는 데이터 누수 방지
- 동일 충전 policy가 학습과 검증에 섞이지 않는 공정한 분할
- Group CV 평균 성능과 fold 간 변동성
- 학습 수명 범위 밖에서의 외삽 가능성
- Feature와 예측 결과의 설명 가능성
- ESS 운영 의사결정에 활용할 수 있는 범위

## 2. Batch별 데이터 역할

모델을 개발하기 전에 각 Batch의 역할을 다음과 같이 고정했습니다.

| 역할 | 데이터 | 라벨 셀 | 사용 목적 |
|---|---|---:|---|
| Train | Batch 1 일부 | 35 | 후보 Feature·모델 비교와 Group CV |
| Valid | Batch 1 Hold-out | 11 | 최종 후보의 마지막 확인 |
| Test | Batch 2 | 39 | 설계 고정 후 외부 성능 평가 1회 |
| Additional | Batch 3 | 44 | EDA와 추가 일반화 한계 확인 |

`cycle_life`가 없는 셀은 목표값을 평균이나 중앙값으로 채우지 않고 지도학습과 성능 평가에서 제외했습니다. 목표값을 임의로 대체하면 실제로 관측되지 않은 수명을 정답처럼 사용하게 되기 때문입니다.

## 3. Policy 단위 데이터 분할

### 3.1 셀 무작위 분할을 사용하지 않은 이유

같은 `policy`로 실험된 셀은 충전속도, 충전 전환 시점, 충전시간 등의 조건을 공유합니다. 같은 policy의 셀이 Train과 Valid에 동시에 들어가면 모델이 이미 본 충전조건과 매우 유사한 셀을 다시 예측하게 되어 성능이 실제보다 좋게 보일 수 있습니다.

이를 방지하기 위해 같은 policy의 모든 셀을 하나의 그룹으로 묶었습니다. 분할 이후 Train과 Valid가 공유하는 policy는 0개입니다.

### 3.2 고정 Policy Hold-out

최초 분할은 다음 설정으로 한 번 수행했습니다.

```python
GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)
```

Batch 1의 23개 policy 중 5개 policy, 11개 셀을 고정 Valid로 분리하고 나머지 18개 policy, 35개 셀을 Train으로 사용했습니다.

### 3.3 Split manifest

각 셀이 어떤 역할을 맡았는지는 `results/modeling/split_manifest_v1.csv`에 저장했습니다. 이후 실행에서는 새로운 분할을 만들지 않고 기존 manifest를 다시 불러옵니다.

Manifest에는 다음 정보가 기록됩니다.

- `batch`
- `cell_id`
- `policy`
- `split`: `train`, `valid`, `test`, `additional`
- `random_state`

이 방식은 결과를 확인한 뒤 유리한 Hold-out으로 바꾸는 것을 방지하고, 동일한 분할을 재현할 수 있게 합니다.

### 3.4 Policy Group Cross-Validation

고정 Valid를 제외한 Batch 1 Train 35개 셀에서는 `GroupKFold(n_splits=5)`를 수행했습니다. 같은 policy의 셀은 항상 같은 fold에 들어가며, 각 fold의 학습 policy와 검증 policy는 겹치지 않습니다.

이는 5회 랜덤 반복 검증이 아니라 **policy를 그룹으로 사용한 5-fold Group CV**입니다. `random_state=42`는 Hold-out을 만든 `GroupShuffleSplit`에만 사용되고, `GroupKFold` 자체는 random state를 사용하지 않습니다.

## 4. 데이터 누수 변수 제거

모델 입력에는 초기 100사이클 시점에 알 수 있는 수치형 정보만 사용했습니다.

| 제외 변수 | 제외 이유 |
|---|---|
| `cycle_life` | 예측하려는 Target 자체 |
| `life_group`, `label_550` | Target에서 파생된 변수 |
| `n_cycles`, `summary_cycles` | 전체 관측 길이로 수명을 직접 암시 |
| knee 관련 변수 | 수명 후반부 전체 곡선이 필요한 미래정보 |
| `cell_id` | 단순 식별자 |
| `batch` | 평가 그룹을 예측 신호로 학습할 위험 |
| 원본 `policy` 문자열 | 모델 입력이 아니라 분할 group으로만 사용 |

## 5. Feature Set 사전 정의

Batch 2 결과를 본 뒤 유리한 변수 조합으로 바꾸지 않도록 세 가지 Feature Set을 미리 고정했습니다.

| Feature Set | 구성 | 비교 목적 |
|---|---|---|
| A | `deltaQ_logvar` | DAY1 EDA에서 확인한 핵심 기준 Feature |
| B | `deltaQ_logvar`, `deltaQ_skew`, `deltaQ_kurtosis` | ΔQ 분포 형태정보의 추가 효과 확인 |
| C | `deltaQ_logvar`, 평균 온도, 평균 충전시간, 첫 단계 C-rate | 보조변수의 추가 효과 확인 |

Feature Set A는 `deltaQ_logvar`가 Batch 1에서 수명과 가장 강하고 명확한 관계를 보였기 때문에 기준 모델로 설정했습니다. B와 C는 변수를 더 추가했을 때 성능과 안정성이 실제로 개선되는지를 확인하는 비교군입니다.

## 6. 후보 모델과 선정 목적

| 후보 모델 | 비교한 이유 |
|---|---|
| Median Baseline | 중앙값 예측보다 학습 모델이 실제로 나은지 확인 |
| Linear Regression | 단일 Feature 관계의 설명과 선형 외삽 |
| Ridge | 여러 Feature 사용 시 L2 규제 효과 확인 |
| Elastic Net | 규제와 변수 선택 효과 확인 |
| Random Forest | 비선형 관계의 이득 확인 |
| Gradient Boosting | 순차적인 비선형 보정 효과 확인 |

Median Baseline을 포함해 복잡한 모델이 단순 기준보다 의미 있게 개선되는지를 확인했습니다. 트리 모델은 비선형 관계를 잘 표현할 수 있지만, 학습 Feature 범위 밖에서 선형적인 외삽을 하기 어렵다는 한계도 함께 고려했습니다.

## 7. 전처리 Pipeline

결측값 대체와 표준화는 모델 학습 전에 전체 데이터에 적용하지 않고 Pipeline 내부에서 수행했습니다.

선형모델 Pipeline은 다음 구조입니다.

```text
SimpleImputer(strategy="median")
    → StandardScaler
    → Linear/Ridge/Elastic Net
```

각 CV fold의 Train 부분에서만 중앙값과 표준화 기준을 학습하고, 해당 fold의 Validation에는 변환만 적용했습니다. 이렇게 해야 Validation의 정보가 전처리 과정에서 Train으로 들어가는 누수를 방지할 수 있습니다.

트리 모델은 변수 크기에 민감하지 않으므로 표준화하지 않고 중앙값 대체만 적용했습니다.

## 8. 후보 하이퍼파라미터

후보 설정은 Batch 2를 평가하기 전에 고정했습니다.

| 모델 | 주요 설정 |
|---|---|
| Linear Regression | `fit_intercept=True`, `positive=False` |
| Ridge | `alpha=1.0` |
| Elastic Net | `alpha=0.01`, `l1_ratio=0.5` |
| Random Forest | `n_estimators=300`, `max_depth=3`, `min_samples_leaf=3` |
| Gradient Boosting | `n_estimators=100`, `learning_rate=0.03`, `max_depth=2` |

Batch 1 Train이 35개 셀인 소표본이므로 트리 깊이를 낮게 설정했습니다. 이 값들은 대규모 Grid Search로 최적값을 찾은 결과가 아니라, 모델군별 특성을 비교하기 위해 미리 정한 보수적인 설정입니다. 최종 Linear Regression에는 `alpha`나 `max_depth`와 같은 핵심 복잡도 하이퍼파라미터가 없어 별도의 Grid Search를 수행하지 않았습니다.

## 9. Batch 1 Group CV 결과

주 평가 지표는 MAPE이며 평균과 표준편차를 함께 비교했습니다.

| 모델 | 대표 Feature Set | Group CV MAPE |
|---|---|---:|
| Median Baseline | A | 21.13 ± 8.38% |
| **Linear Regression** | **A** | **7.60 ± 1.57%** |
| Elastic Net | C | 8.80 ± 1.75% |
| Ridge | B | 8.90 ± 2.69% |
| Gradient Boosting | A | 9.42 ± 4.15% |
| Random Forest | A | 9.66 ± 3.85% |

Linear Regression과 Feature Set A 조합이 평균 MAPE가 가장 낮았고 fold별 변동성도 작았습니다. 복잡한 모델은 평균 성능을 개선하지 못했으며 트리 모델은 fold 간 변동성도 컸습니다.

## 10. 최종 모델 선정

최종 Pipeline은 **`deltaQ_logvar` 하나를 사용하는 Linear Regression**입니다.

선정 근거는 다음과 같습니다.

1. Batch 1 Group CV 평균 MAPE가 가장 낮았습니다.
2. CV 표준편차가 작아 fold가 바뀌어도 비교적 안정적이었습니다.
3. 단일 Feature이므로 Ridge·Elastic Net의 규제 이점이 크지 않았습니다.
4. `deltaQ_logvar` 변화에 따른 예상수명 변화를 계수로 설명할 수 있습니다.
5. Random Forest보다 학습 Target 범위 밖으로 선형 추세를 연장할 수 있습니다.
6. Feature 수와 모델 복잡도가 작아 소표본에서 과적합 위험을 줄일 수 있습니다.

최고 성능만 보고 선택한 것이 아니라 성능, 안정성, 설명 가능성, 외삽 특성이 분석 목적과 맞는지를 함께 고려했습니다.

## 11. 최종 성능

모델을 선택한 뒤 고정 Valid를 한 번 확인하고, 최종 Pipeline을 Batch 1 전체 46개 셀로 재학습한 다음 Batch 2의 39개 라벨 셀을 한 번 평가했습니다.

| 구분 | MAPE | 해석 |
|---|---:|---|
| Train (Batch 1 Group CV) | 7.60 ± 1.57% | 후보 선택 기준 |
| Valid (Batch 1 Hold-out) | 11.63% | 보지 못한 5개 policy 확인 |
| Test (Batch 2) | 31.53% | 외부 배치 최종 평가 |
| Gap (Train–Valid) | +4.03%p | 내부 일반화 성능 저하 |
| Gap (Valid–Test) | +19.90%p | 배치 간 일반화 성능 저하 |
| Gap (Target–Test) | +22.43%p | 원논문 9.1% 대비 차이 |

Batch 2의 보조 지표는 다음과 같습니다.

- MAE: 155.59사이클
- RMSE: 169.56사이클
- R²: 0.402

## 12. Batch 2 오류 분석

Batch 1 전체의 최소 수명은 534사이클이었습니다. Batch 2 평가 셀 39개 중 30개, 즉 76.9%가 이보다 짧았습니다. 최종 모델은 이 30개 셀을 모두 실제보다 길게 예측했습니다.

| 구간 | 결과 |
|---|---:|
| Batch 1 최소 수명보다 짧은 Batch 2 셀 | 30/39개, 76.9% |
| 해당 셀의 과대예측 | 30/30개, 100% |
| 단수명 셀 평균 APE | 36.73% |
| 나머지 셀 평균 APE | 14.20% |

이는 모델이 학습하지 못한 단수명 영역에서 수명을 낙관적으로 예측하는 외삽 문제를 보여줍니다. ESS에서는 위험 셀의 수명을 실제보다 길게 판단하는 것이 안전 문제로 이어질 수 있습니다.

따라서 현재 모델은 정확한 교체 시점이나 보증수명을 단독으로 결정하기보다 다음 용도로 한정하는 것이 적절합니다.

- 초기 열화가 큰 셀의 선별
- 추가 장기시험 대상의 우선순위 결정
- 입고 검사와 품질관리 보조
- 유지보수 대상의 초기 위험도 확인

## 13. 원논문 성능과의 비교

원논문의 550사이클은 Classification에서 장수명과 단수명을 나누는 Target 기준이며, Regression의 입력 관측 구간이 아닙니다. 원논문 회귀와 본 분석은 모두 초기 100사이클 정보를 사용합니다.

따라서 원논문 MAPE 9.1%와 본 분석의 31.53% 차이를 관측 사이클 수 차이로 설명할 수 없습니다. 다음 요인을 함께 고려해야 합니다.

- 원논문과 본 분석의 Feature 구성 차이
- 셀 제외 및 데이터 정제 기준 차이
- Train·Test 구성과 policy 분할 방식 차이
- Batch 2에 단수명 셀이 집중된 Target 분포 이동
- `deltaQ_logvar` 분포 이동
- Batch 1의 소표본과 policy별 반복 부족
- 단수명 셀에서 비율오차가 커지는 MAPE의 특성

각 요인이 성능 저하에 기여한 정도는 현재 실험만으로 분리할 수 없으므로 하나의 원인으로 단정하지 않습니다.

## 14. 팀원 모델과 비교할 때 확인할 기준

두 모델은 다음 조건을 먼저 확인한 뒤 비교해야 합니다.

1. 동일한 수명 라벨과 셀 목록을 사용했는가?
2. 초기 100사이클 이후의 미래정보가 Feature에 포함되지 않았는가?
3. 같은 policy가 Train과 Valid에 동시에 포함되지 않았는가?
4. 전처리가 각 CV Train fold 내부에서만 학습됐는가?
5. CV 평균뿐 아니라 표준편차도 함께 비교했는가?
6. Hold-out policy가 다르다는 사실을 고려했는가?
7. 외삽과 과대예측 위험을 확인했는가?
8. 모델 선택 이유가 Feature 특성과 분석 목적에 맞는가?

두 사람의 Hold-out policy가 다르면 Valid MAPE는 시험 난이도가 서로 다르므로 숫자만으로 직접 우열을 판단하면 안 됩니다. Batch 2 Test 셀 목록과 전처리 정의가 같다면 Test MAPE는 비교할 수 있지만, Batch 2 결과만 보고 최종 모델을 선택하면 Batch 2가 사실상 모델 선택 데이터가 됩니다.

이미 두 모델의 Batch 2 결과를 확인했다면 최종 선택은 다음을 중심으로 판단합니다.

- Batch 1 Group CV 평균과 표준편차
- 고정 Valid에서의 성능 저하 정도
- 데이터 누수 방지 수준
- Feature의 도메인 의미
- 모델 복잡도와 설명 가능성
- 새로운 수명 범위로의 외삽 가능성
- ESS 운영에서의 과대예측 위험

최종 보고서에는 다음과 같이 기록할 수 있습니다.

> 각자 사전에 고정한 Pipeline의 Batch 2 결과를 비교했으며, 최종 Pipeline은 Batch 1 내부 검증의 성능과 안정성, 데이터 누수 방지, Feature의 도메인 의미 및 모델의 설명 가능성을 중심으로 결정했다. Batch 2 평가 후에는 현재 Test 성능을 개선하기 위한 Feature·모델·하이퍼파라미터 변경을 수행하지 않았다.

## 15. 우리 모델 비교용 요약표

| 비교 항목 | 우리 모델 |
|---|---|
| 과제 | Regression |
| 핵심 Feature | `deltaQ_logvar` 1개 |
| 최종 모델 | Linear Regression |
| Hold-out | Batch 1의 5개 policy, 11개 셀 |
| Group CV | Batch 1 Train 35셀, 18개 policy, GroupKFold(5) |
| Train CV MAPE | 7.60 ± 1.57% |
| Valid MAPE | 11.63% |
| Batch 2 Test MAPE | 31.53% |
| 주요 장점 | 단순성, 설명 가능성, 선형 외삽 |
| 주요 한계 | Batch 2 단수명 영역 과대예측 |
| 활용 범위 | 초기 위험 셀 선별과 추가 시험 우선순위 결정 보조 |
