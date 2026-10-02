# ESS 배터리 수명 예측

초기 100사이클 이내의 충·방전 특성으로 배터리 `cycle_life`를 예측하여 장기 수명시험과 초기 위험 셀 선별을 보조하는 회귀 프로젝트입니다. 모델과 Feature는 Batch 1에서만 선택하고, Batch 2는 설계를 고정한 뒤 한 번 평가했습니다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., 2019)
- 학습 데이터: Batch 1 (`2017-05-12`)
- 최종 평가 데이터: Batch 2 (`2018-02-20`)
- 추가 분석 데이터: Batch 3 (`2018-04-12`, EDA·한계 확인만 사용)
- 과제 유형: Regression
- Target: 연속형 `cycle_life`
- 주 지표: MAPE
- 보조 지표: MAE, RMSE, R²
- 원논문 비교 기준: Regression MAPE 9.1%

| 역할 | 데이터 | 사용 방법 |
|---|---|---|
| Train | Batch 1 | policy 기반 5-fold Group CV |
| Valid | Batch 1 | policy 5종 고정 Hold-out |
| Test | Batch 2 | 모델 고정 후 최종 평가 1회 |
| Additional | Batch 3 | EDA와 외삽 한계 확인, 모델 평가 제외 |

## 파일 구조

```text
├── README.md
├── requirements.txt
├── data/
│   └── README.md
├── notebooks/
│   ├── 01_DAY1_EDA.ipynb
│   └── 02_DAY2_Modeling.ipynb
├── src/
│   ├── battery_eda.py
│   ├── modeling_utils.py
│   └── preprocess.py
├── outputs/
│   ├── model_comparison.csv
│   ├── final_predictions_batch2.csv
│   ├── performance_report.csv
│   └── figures/
└── results/
    └── modeling/
```

원본 `.mat` 파일과 로컬 캐시는 GitHub에 올리지 않습니다. `results/modeling/`에는 분할·모델 설정 기록과 상세 분석 결과가 저장됩니다.

## 환경 설정

```bash
git clone https://github.com/jihoo1018/data-mini-project.git
cd data-mini-project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

실행 순서:

1. `notebooks/01_DAY1_EDA.ipynb`
2. `notebooks/02_DAY2_Modeling.ipynb`

주요 환경은 Python 3.11.15, NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1, scikit-learn 1.9.1이며 `random_state=42`를 사용했습니다.

## EDA 핵심 발견

- **Cycle Life 분포:** 수명 중앙값은 Batch 1 858.5, Batch 2 472.0, Batch 3 1,005.5사이클이었습니다. 550사이클 미만 비율은 각각 2.2%, 76.9%, 2.3%로 배치 차이가 컸습니다.
- **열화 곡선:** 초기 100사이클의 방전용량 자체는 셀 간 차이가 작았습니다. Knee point는 수명 후반에 확인되므로 초기 예측 Feature로 사용하면 미래정보 누수가 발생합니다.
- **ΔQ(V):** Batch 1에서 `deltaQ_logvar`와 수명은 강한 음의 관계를 보였습니다(Pearson `r=-0.886`). Batch 2·3에서의 관계는 모델 선택이 끝난 뒤 배치 일관성을 해석하는 근거로만 사용했습니다.
- **충전 속도와 조건:** 평균 C-rate·온도·충전시간은 Batch 1에서 관계가 보여도 다른 배치에서 크기나 부호가 달라 핵심 Feature에서 제외했습니다.
- **모델링 시사점:** Batch 2 라벨 셀의 76.9%가 Batch 1 최소 수명 534사이클보다 짧아 외삽 오차가 예상됐습니다.

## Modeling

### Feature Engineering 전략

최종 입력은 Feature Set A의 `deltaQ_logvar` 하나입니다.

```python
selected_features = ["deltaQ_logvar"]
```

`deltaQ_logvar`는 초기 방전용량 곡선 변화 `Q100(V) - Q10(V)`의 분산을 `log10`으로 변환한 값입니다. 초기 열화의 크기를 대표하고 Batch 1에서 수명과 강한 관계를 보여 선택했습니다. 이미 로그 변환된 값이므로 Pipeline에서 다시 로그 변환하지 않았습니다.

제외 원칙:

- Target·Target 파생값, `n_cycles`, knee 계열, 100사이클 이후 변수: 데이터 누수 위험
- `cell_id`, `batch`: 식별자 및 평가 그룹
- 원본 `policy`: 입력이 아니라 분할용 group으로만 사용
- ΔQ 중복 변수: `deltaQ_logvar`와 정보가 겹침
- 온도·충전시간·C-rate: Batch 1 Group CV에서 안정적인 개선이 없음
- 수명 라벨 결측 셀: Target을 대체하지 않고 지도학습·평가에서 제외

### 후보 모델 및 하이퍼파라미터

| 후보 | 비교 목적 | 주요 설정 |
|---|---|---|
| Median Baseline | 학습의 최소 기준 | `strategy="median"` |
| Linear Regression | 단일 Feature 관계의 해석과 외삽 | `fit_intercept=True`, `positive=False` |
| Ridge | 다변수 규제 효과 | `alpha=1.0` |
| Elastic Net | 규제와 변수 선택 효과 | `alpha=0.01`, `l1_ratio=0.5` |
| Random Forest | 비선형 관계 비교 | `max_depth=3`, `min_samples_leaf=3` |
| Gradient Boosting | 순차적 비선형 보정 비교 | `learning_rate=0.03`, `max_depth=2` |

후보 설정은 Batch 2 평가 전에 고정했습니다. 소표본에서 각 모델군의 특성을 비교하기 위한 보수적 설정이며, 대규모 Grid Search로 최적값을 찾았다고 주장하지 않습니다. 최종 Linear Regression에는 `alpha`나 `max_depth`처럼 복잡도를 조절하는 핵심 하이퍼파라미터가 없습니다.

### 최종 모델과 선정 이유

최종 모델은 **Linear Regression + Feature Set A (`deltaQ_logvar`)**입니다.

- Batch 1 Group CV MAPE가 7.60±1.57%로 후보 중 가장 낮고 안정적이었습니다.
- 변수가 하나이므로 Ridge·Elastic Net의 규제 이점이 크지 않았습니다.
- 트리 모델보다 계수 해석과 학습 범위 밖 외삽이 용이했습니다.
- 복잡한 모델과 차이가 작으면 더 단순하고 해석 가능한 모델을 선택한다는 규칙을 Hold-out 확인 전에 고정했습니다.

### 학습 및 평가 전략

1. Batch 1/2 역할을 모델링 전에 고정하고 split manifest로 저장
2. Batch 1을 policy 단위 Train 35셀·Valid 11셀로 분리
3. Train의 18개 policy에서 `GroupKFold(n_splits=5)` 수행
4. 중앙값 대체와 표준화를 각 CV Train fold 내부에서만 fit
5. Batch 1 CV 결과만으로 Feature와 모델 선택
6. 고정 Hold-out을 한 번 확인한 뒤 Batch 1 전체 46셀로 재학습
7. Batch 2 라벨 39셀을 한 번 평가하고 이후 모델·Feature를 변경하지 않음

### 우리 조의 분석 특징

- 무작위 셀 분할 대신 `policy` 단위 Hold-out과 Group CV를 사용해 동일 충전정책의 누수를 통제했습니다.
- 내부 점수만 가장 좋은 복잡한 모델보다 초기 Feature의 의미, 외삽 가능성, 설명 가능성을 함께 고려해 단순 선형회귀를 선택했습니다.
- 낮은 Batch 2 성능을 숨기지 않고 단수명 셀 과대예측을 ESS 안전 리스크와 적용 범위 제한으로 연결했습니다.

## 성능 결과

| 구분 | MAPE (%) | 비고 |
|---|---:|---|
| Train (Batch 1 CV) | 7.60 ± 1.57 | Group CV |
| Valid (Batch 1 Hold-out) | 11.63 | 고정 policy Hold-out |
| Test (Batch 2) | 31.53 | 최종 평가 1회 |
| Gap (Train–Valid) | +4.03%p | Valid − Train; 양수: 과적합 의심 |
| Gap (Valid–Test) | +19.90%p | Test − Valid; 양수: 배치 일반화 저하 |
| Gap (Target–Test) | +22.43%p | Test − 9.1; 양수: 원논문보다 낮은 성능 |

Batch 2 보조 지표는 MAE 155.59 cycles, RMSE 169.56 cycles, R² 0.402입니다. 원논문의 550사이클은 분류의 장·단수명 기준이며 회귀 관측 구간이 아닙니다. 원논문 회귀와 본 분석은 모두 초기 100사이클 정보를 사용하므로, Target 9.1%와의 차이는 Batch 2의 단수명 외삽, 핵심 Feature 분포 이동, Batch 1의 소표본, Feature 구성과 데이터 정제·분할·평가 범위 차이로 해석했습니다.

성능 그래프:

- `outputs/figures/actual_vs_predicted_v1.png`
- `outputs/figures/residual_plot_v1.png`
- `outputs/figures/model_cv_comparison_v1.png`
- `outputs/figures/batch2_cell_ape_v1.png`

## DAY1 전략 대비 실제 결과

| DAY1 전략 및 예상 | DAY2 실제 결과 | 판단과 원인 |
|---|---|---|
| `deltaQ_logvar` 단독 모델이 보조변수 추가 모델보다 안정적일 것이다 | Linear Regression이 7.60±1.57%로 가장 우수 | 적중: 보조변수의 배치 관계가 불안정하고 소표본에서 추가 실익이 없었음 |
| policy 단위 Group CV로 누수를 방지한다 | GroupKFold와 policy Hold-out 적용 | 적중: 같은 policy가 Train·Valid에 섞이지 않음 |
| 선형모델이 트리 모델보다 외삽에 유리할 것이다 | 선형회귀의 CV 평균과 변동성이 더 낮았음 | 적중: 트리 모델은 외삽과 소표본에 불리했음 |
| Batch 2 단수명 영역에서 외부 성능이 저하될 것이다 | Test MAPE 31.53%, Valid-Test Gap +19.90%p | 예상 위험 발생: 76.9%가 Batch 1 최소 수명 아래였음 |
| 원논문 MAPE 9.1%에 가까운 성능을 목표로 한다 | Target-Test Gap +22.43%p | 미달: 배치 이동, 외삽, 소표본, 재현 조건 차이 |

## 오류 분석

- Batch 2 라벨 셀 39개 중 30개(76.9%)가 Batch 1 최소 수명보다 짧았고, 30개 모두 실제보다 길게 예측됐습니다.
- 단수명 셀의 평균 APE는 36.73%, 나머지 셀은 14.20%였습니다.
- Batch 1–2 Target KS 통계량은 0.769, `deltaQ_logvar` KS 통계량은 0.591로 분포 이동이 컸습니다.
- `deltaQ_logvar` 하나만으로 배치 효과와 셀 조건 차이를 모두 설명하지 못한 것이 주요 한계입니다.

개선 방향은 단수명 학습 셀 확충, 다양한 제조 배치의 외부 검증, 예측 불확실성 제공입니다. Batch 2를 이미 확인했으므로 이 개선안으로 현재 Test에 다시 맞추지 않고 후속 실험으로 분리합니다.

## ESS 도메인 해석

현재 모델은 Batch 2 MAPE 31.53%, MAE 약 156사이클로 정확한 교체 시점이나 보증수명을 결정하기에는 부족합니다. 다만 초기 100사이클만으로 열화 위험을 조기에 확인하므로 다음 의사결정의 보조 수단으로 활용 가능성이 있습니다.

- 추가 장기시험이 필요한 셀의 우선순위 선정
- 초기 열화가 큰 셀의 선별
- 배터리 입고 검사와 품질관리 보조
- 유지보수 대상의 사전 위험도 분류

특히 단수명 셀을 모두 과대예측한 결과는 실제 ESS에서 위험 셀의 수명을 낙관할 가능성을 뜻합니다. 실제 적용 전에는 단수명 데이터 확충, 배치별 재검증과 안전 중심의 과대예측 기준이 필요합니다.

## 한계 및 개선 방향

- Batch 1은 라벨 셀 46개, policy 23개이며 policy당 반복은 1–3개입니다.
- Batch 2 전체 47셀 중 수명 결측 8개는 평가에서 제외했습니다.
- Batch 3은 수명 결측 2개와 최대 1,935사이클의 초장수명 셀을 포함하지만 최종 평가에 사용하지 않았습니다.
- `newstructure`의 정확한 의미가 확인되지 않아 Feature로 사용하지 않았습니다.
- 원논문과 데이터 정정본, 전처리, Feature 계산, 분할 방식과 평가 범위가 완전히 같다고 보장할 수 없습니다.
- 상관계수와 모델 계수는 예측적 연관성이며 전기화학적 인과를 직접 증명하지 않습니다.
- 실제 배포 전에는 제조 배치·온도·운영조건이 다른 데이터로 외부 검증하고, 입력 분포 이동과 모델 성능을 지속적으로 모니터링해야 합니다.

## 참고문헌

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성

- 문지후·정승우: 각자 독립적인 EDA·모델링 실험 수행, 결과 비교 및 최종 보고서 공동 작성
