# 초기 충·방전 특성을 활용한 배터리 수명 예측

초기 100사이클 안에서 계산 가능한 특성으로 배터리의 `cycle_life`를 예측하는 회귀 프로젝트입니다. 모델 선택은 Batch 1만 사용하고, Batch 2는 설계를 모두 고정한 뒤 한 번만 평가했습니다. Batch 3는 EDA와 한계 확인에만 사용했습니다.

## 1. 분석 전략

| 역할 | 데이터 | 사용 방법 |
|---|---|---|
| Train | Batch 1 | policy 기반 Group CV |
| Valid | Batch 1 | policy 단위 고정 Hold-out |
| Test | Batch 2 | 최종 모델 고정 후 1회 평가 |
| Additional | Batch 3 | EDA·외삽 한계 확인, 모델 평가 제외 |

- 목표변수: 연속형 `cycle_life`
- 문제 유형: Regression
- 주 지표: MAPE
- 보조 지표: MAE, RMSE, R²
- 고정 난수: `random_state=42`
- 원논문 비교 기준: Regression MAPE 9.1%

## 2. Feature 전략

### 최종 입력 Feature

```python
selected_features = ["deltaQ_logvar"]
```

`deltaQ_logvar`는 초기 방전용량 곡선 변화 ΔQ(V)의 분산을 로그로 변환한 값입니다. Batch 1에서 수명과 강한 음의 상관을 보였고(Pearson `r=-0.886`), 초기 열화 정도를 대표하는 변수로 사용했습니다. 이미 로그 변환된 값이므로 Pipeline에서 다시 로그 변환하지 않습니다.

### 제외 Feature와 이유

| 변수 | 제외 이유 |
|---|---|
| `cycle_life`, `life_group`, `label_550` | Target 또는 Target 파생값으로 데이터 누수 발생 |
| `n_cycles`, `summary_cycles` | 전체 관측 길이로 수명을 직접 암시 |
| `knee_cycle_proxy`, `knee_fraction_proxy` | 수명 후반부 곡선과 EOL 정보가 필요 |
| `cell_id`, `batch` | 식별자·평가 그룹이며 예측변수가 아님 |
| 원본 `policy` 문자열 | 분할용 group으로만 사용 |
| 100사이클 이후 계산 변수 | 실제 초기 예측 시점에 알 수 없는 미래정보 |
| ΔQ 중복 변수 | `deltaQ_logvar`와 정보가 크게 겹쳐 중복 입력을 피함 |
| 온도·충전시간·C-rate 보조변수 | Batch 1 CV에서 단일 Feature 모델을 안정적으로 개선하지 못함 |

수명 라벨이 없는 셀은 X 결측치처럼 대체하지 않고 지도학습·성능평가에서 제외했습니다.

## 3. 후보 모델과 선정 근거

| 후보 | 사용 이유 |
|---|---|
| Median Baseline | 학습의 최소 기준점 |
| Linear Regression | 단일 핵심 Feature의 방향과 계수를 설명하기 쉬움 |
| Ridge / Elastic Net | 복수 Feature에서 규제 효과 확인 |
| Random Forest | 비선형 관계 비교용 |
| Gradient Boosting | 순차적 비선형 보정 비교용 |

Batch 1은 라벨 셀이 46개뿐이고 Batch 2의 단수명 영역은 Batch 1 Target 범위 밖에 있습니다. 따라서 복잡한 트리 모델의 과적합·변동성과 외삽 한계를 함께 고려했습니다. CV MAPE가 최저 후보와 1%p 이내면 더 단순하고 해석 가능한 모델을 우선한다는 규칙을 Hold-out 확인 전에 고정했습니다.

최종 선택은 **Linear Regression + Feature Set A (`deltaQ_logvar`)**입니다.

## 4. 데이터 누수 방지 Pipeline

1. Batch 1/2 역할을 모델링 전에 고정하고 `split_manifest_v1.csv`로 저장
2. Batch 1 Train/Valid를 policy 단위로 분리
3. Batch 1 Train 내부에서 `GroupKFold(n_splits=5)` 수행
4. 모든 전처리를 Pipeline 내부에서 fold의 Train에만 fit
5. 선형모델 Pipeline: Median Imputer → StandardScaler → Model
6. Target·미래정보·식별자·원본 policy를 X에서 제거
7. Hold-out 전에 후보를 JSON으로 잠금
8. Batch 2 전에 Feature·모델·전처리·하이퍼파라미터를 다시 잠금
9. Batch 1 전체로 재학습 후 Batch 2를 1회 평가

Batch 2 결과를 확인한 뒤에는 Feature, 모델 또는 하이퍼파라미터를 변경하지 않았습니다. 노트북 재실행 시에도 최초 Batch 2 평가 결과를 불러오도록 보호했습니다.

## 5. 최종 성능

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 7.60 ± 1.57 | Group CV |
| Valid (Batch 1 Hold-out) | 11.63 | 고정 policy Hold-out |
| Test (Batch 2) | 31.53 | 최종 평가 1회 |
| Gap (Train–Valid) | +4.03%p | Valid − Train; 양수: 과적합 의심 |
| Gap (Valid–Test) | +19.90%p | Test − Valid; 양수: 배치 일반화 저하 |
| Gap (Target–Test) | +22.43%p | Test − 9.1; 양수: 원논문보다 낮은 성능 |

Batch 2 보조 지표:

- MAE: 155.59 cycles
- RMSE: 169.56 cycles
- R²: 0.402

## 6. 주요 결과 해석

- `deltaQ_logvar`가 0.1 증가할 때 최종 선형모델은 수명을 약 43.3사이클 짧게 예측합니다.
- Batch 2 라벨 셀 39개 중 30개(76.9%)가 Batch 1 최소 수명 534사이클보다 짧습니다.
- 이 30개는 모두 실제보다 길게 예측됐습니다.
- 단수명 셀의 평균 APE는 36.73%, 나머지 셀은 14.20%였습니다.
- Batch 1–2 Target KS 통계량은 0.769, `deltaQ_logvar` KS 통계량은 0.591로 분포 이동이 큽니다.
- 복잡한 모델은 선형회귀보다 CV 평균 MAPE가 낮지 않았고, 트리 모델은 fold 변동성도 컸습니다.
- 원논문 9.1%와의 차이는 배치 이동, 단수명 외삽, 소표본, Feature·전처리·평가 범위 차이가 함께 작용한 것으로 해석합니다. 개별 요인의 인과 기여도는 현재 실험으로 분리할 수 없습니다.

## 7. 제출 산출물

GitHub 제출용 핵심 파일은 다음과 같습니다.

```text
README.md
requirements.txt
.gitignore
data/README.md
notebooks/01_DAY1_EDA.ipynb
notebooks/02_DAY2_Modeling.ipynb
outputs/model_comparison.csv
outputs/final_predictions_batch2.csv
outputs/performance_report.csv
outputs/figures/
```

`results/modeling/`에는 분할 manifest, 모델·Feature 잠금 기록과 상세 분석 결과가 추가로 보관됩니다.

## 8. 성능 그래프

- `outputs/figures/actual_vs_predicted_v1.png`
- `outputs/figures/residual_plot_v1.png`
- `outputs/figures/model_cv_comparison_v1.png`
- `outputs/figures/batch2_cell_ape_v1.png`

그래프에서 Batch 1 Hold-out과 Batch 2를 구분했으며, 실제값–예측값 그래프에는 `y=x`, 잔차 그래프에는 0 기준선을 표시했습니다. 모델 비교 그래프는 Group CV 평균 MAPE와 표준편차를 함께 보여줍니다.

## 9. 한계

- Batch 1은 라벨 셀 46개, policy 23개의 소표본입니다.
- policy당 셀은 1–3개로 반복 수가 적습니다.
- Batch 2 전체 47셀 중 수명 결측 8개는 평가에서 제외했습니다.
- Batch 3은 라벨 44셀, 수명 결측 2셀이며 최대 수명이 1,935사이클입니다.
- `newstructure`의 정확한 의미가 확인되지 않아 Feature로 사용하지 않았습니다.
- 원논문과 데이터 정정본, 전처리, Feature 계산, 분할 방식과 평가 범위가 완전히 같다고 보장할 수 없습니다.
- 상관계수와 모델 계수는 예측적 연관성이며 전기화학적 인과를 직접 증명하지 않습니다.

## 10. 실행 방법

원본 데이터 준비 방법은 `data/README.md`를 참고합니다. 원본 `.mat` 파일과 캐시는 GitHub에 올리지 않습니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

jupyter lab
```

실행 순서:

1. `notebooks/01_DAY1_EDA.ipynb`
2. `notebooks/02_DAY2_Modeling.ipynb`

모델링 노트북은 `results/battery_eda/cell_features.csv`를 자동으로 찾으며, 결과는 `results/modeling/`에 저장합니다.

## 11. 환경 기록

- Python 3.11.15
- NumPy 2.4.6
- pandas 3.0.6
- SciPy 1.17.1
- Matplotlib 3.11.2
- seaborn 0.13.2
- scikit-learn 1.9.1
- h5py 3.16.0
- 프로젝트 `random_state=42`

세부 설정과 재현 가능한 수치 근거는 `results/modeling/*.json`, `*.csv` 및 실행 완료 노트북에 기록돼 있습니다.
