# ESS 배터리 수명 예측

초기 100사이클 이내의 충·방전 특성으로 배터리 `cycle_life`를 예측하여 장기 수명시험과 초기 위험 셀 선별을 보조하는 회귀 프로젝트입니다. 모델과 Feature는 Batch 1에서만 선택하고, Batch 2는 설계를 고정한 뒤 한 번 평가했습니다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., 2019)
- 학습 데이터: Batch 1 (`2017-05-12`)
- 최종 평가 데이터: Batch 2 (`2018-02-20`)
- 추가 분석 데이터: Batch 3 (`2018-04-12`, EDA·한계 확인만 사용)
- 과제 유형: Regression (과제 안내의 회귀/분류 택1 중 **회귀** 선택. 분류는 본 분석이 아니라 원논문 성능과 견주기 위한 참고로만 `성능 결과` 하단에 정리)
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
│   └── README.md                    # 데이터 사용법: 원본 받기 → 캐시 만들기 → 노트북에서 읽기
├── notebooks/
│   ├── 01_DAY1_EDA.ipynb            # EDA (메인)
│   ├── 01_DAY1_EDA_sub.ipynb        # EDA를 다른 방식으로 시도해 본 서브 파일
│   ├── 02_hypothesis_H1_H2.ipynb    # 모델링 전 가설 검증용 서브 파일
│   └── 02_DAY2_Modeling.ipynb       # 모델링 (메인, 최종 파이프라인)
├── src/
│   ├── battery_eda.py
│   ├── modeling_utils.py
│   └── preprocess.py
├── version_statement.md              # DAY별 진행 내용 요약(버전 관리용 작업 기록)
├── outputs/
│   ├── model_comparison.csv
│   ├── final_predictions_batch2.csv
│   ├── performance_report.csv
│   └── figures/
└── results/
    ├── battery_eda/                  # EDA 산출물(cell_features.csv 등, 모델링 입력)
    └── modeling/                     # 분할·모델 설정, 예측, 성능 결과
```

`version_statement.md`는 DAY 1·DAY 2에 진행한 내용(한 일, 결과·결정)을 표 한 장씩으로 요약한 버전 관리용 작업 기록 파일입니다.

원본 `.mat` 파일과 로컬 캐시는 GitHub에 올리지 않습니다. `data/README.md`는 데이터를 어떻게 준비하고 어떤 순서로 돌려야 하는지(원본 `.mat` 받기, `src/preprocess.py`로 캐시 만들기, 노트북에서 읽는 방법)를 설명하는 안내 파일입니다. `results/modeling/`에는 분할·모델 설정 기록과 상세 분석 결과가 저장됩니다.

노트북 역할:

- **메인**: `01_DAY1_EDA.ipynb`(EDA), `02_DAY2_Modeling.ipynb`(모델링). 최종 결과와 성능 표는 모두 이 두 파일 기준입니다.
- **서브**: `01_DAY1_EDA_sub.ipynb`는 같은 EDA를 다른 방식으로 시도해 본 파일, `02_hypothesis_H1_H2.ipynb`는 모델링 전에 가설(H1: feature 1개로 충분한가, H2: 랜덤 CV가 낙관적인가)을 검증하기 위한 파일입니다. 서브 파일은 실행하지 않아도 메인 노트북 결과를 재현할 수 있습니다.

노트북별 필요한 데이터:

| 노트북 | 읽는 데이터 |
|---|---|
| `01_DAY1_EDA.ipynb` | 원본 `.mat`(`data/`) → `results/battery_eda/cell_features.csv` 등 생성 |
| `02_DAY2_Modeling.ipynb` | `results/battery_eda/cell_features.csv`(저장소에 포함) — 원본 `.mat` 없이 실행 가능 |
| `01_DAY1_EDA_sub.ipynb`, `02_hypothesis_H1_H2.ipynb` | `data/cache/`(`src/preprocess.py`로 생성) |

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

1. `notebooks/01_DAY1_EDA.ipynb` (원본 `.mat`가 있을 때)
2. `notebooks/02_DAY2_Modeling.ipynb` (저장소의 `results/battery_eda/cell_features.csv`만으로 실행 가능하므로 `.mat`가 없으면 1번은 건너뛰어도 됩니다)

## EDA 핵심 발견

세 배치의 수명 분포, 열화 곡선, ΔQ(V), 충전 조건을 비교했습니다. 아래 수치는 `01_DAY1_EDA.ipynb`, 서브 파일 `01_DAY1_EDA_sub.ipynb`·`02_hypothesis_H1_H2.ipynb`의 출력에서 확인한 값입니다.

### Cycle Life 분포
- 수명 중앙값은 Batch 1 858.5, Batch 2 472.0, Batch 3 1,005.5사이클입니다. 550사이클 미만 비율은 각각 2.2%, 76.9%, 2.3%입니다.
- Batch 2는 392~514사이클(30셀)과 777~1,186사이클(9셀) 두 덩어리로 나뉘며 Batch 1(534~1,227)과 분포가 다릅니다.
- **핵심 발견:** 배치마다 수명 분포가 크게 달라, Batch 2에는 Batch 1 최소 수명보다 짧은 셀이 76.9%나 있다(외삽 위험).

### 열화 곡선 분석
- 열화는 수명 후반에 가속하고 knee는 수명의 약 71% 지점(Batch 1/2/3: 0.71/0.73/0.76)에서 나타납니다.
- 초기 방전용량의 상승은 Batch 2에서만 뚜렷합니다(사이클 20~100 기울기가 양(+)인 셀 비율 Batch 2 64%, Batch 1 7%, Batch 3 0%).
- **핵심 발견:** knee는 수명 후반에야 보여 초기 100사이클 예측에는 쓸 수 없고(미래정보 누수), 초기 용량은 배치에 따라 양상이 달라 Feature로 부적합하다.

### ΔQ(V) 곡선 분석
- Cycle 100 − Cycle 10의 방전용량 차이 곡선에서 단수명 셀의 ΔQ가 더 깊습니다(약 −0.06Ah 대 −0.01Ah).
- `deltaQ_logvar`와 수명의 상관은 Batch 1/2/3에서 −0.89/−0.90/−0.70으로 유지됩니다(Batch 3 약화는 수명 1,600사이클 이상 초장수명 5셀 영향).
- **핵심 발견:** ΔQ 분산(`deltaQ_logvar`)이 세 배치에서 일관되게 수명과 강한 음의 관계를 보이는 유일한 핵심 Feature다.

### 충전 속도(C-rate)와 수명의 관계
- 평균 C-rate와 수명의 상관이 Batch 1 −0.87에서 Batch 2 −0.25, Batch 3 −0.29로 배치마다 달라집니다.
- 같은 충전 정책도 배치마다 수명이 달랐습니다(공통 정책의 수명 753 → 484, 744 → 400).
- **핵심 발견:** 충전 조건은 Batch 1에서만 강해 보일 뿐 다른 배치로 일반화되지 않아 Feature에서 제외했다.

### 추가 확인한 내용
- **상관 구조(Q5):** ΔQ 계열만 세 배치에서 부호와 크기가 안정적이었습니다. 평균 온도는 부호가 바뀌고(−0.48 → +0.42), 충전시간은 +0.58에서 +0.09로 사라집니다. `deltaQ_logvar`는 다른 ΔQ 변수·평균 C-rate와 서로 겹쳐(0.99, 0.92) 1개만 쓰는 것으로 정리했습니다.
- **랜덤 CV는 낙관적(H2, `02_hypothesis_H1_H2.ipynb`):** 같은 충전 정책의 셀이 train/valid에 섞이면 MAPE가 정책 단위 CV보다 0.65~1.33%p 좋게 나와, policy 단위 분할을 쓴 근거로 삼았습니다.
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
- 트리 모델보다 계수 해석과 학습 범위 밖 외삽이 용이했습니다. 이는 별도 외삽 검증으로 확인했습니다(아래).
- 복잡한 모델과 차이가 작으면 더 단순하고 해석 가능한 모델을 선택한다는 규칙을 Hold-out 확인 전에 고정했습니다.

### 외삽 검증과 시도했으나 채택하지 않은 것

아래는 `02_DAY2_Modeling.ipynb` 13.3에 정리한 **별도 분석**입니다. 같은 모델·Feature로 독립 구현해 얻은 값이며 검증 분할의 세부(Hold-out 정책 5종 중 2종)가 달라 이 README의 성능 표와 직접 비교하지 않고 방향만 참고합니다.

- **외삽 검증:** Batch 1에서 수명이 가장 짧은 25% 또는 가장 긴 25%를 통째로 test로 빼고 학습했을 때 MAPE(짧은 25% / 긴 25%)는 선형 17.5% / 12.2%, Random Forest 22.7% / 21.4%, Gradient Boosting 23.2% / 22.4%였습니다. 트리의 예측이 학습 범위 안에 갇혀(긴 구간 실제 917~1,227인데 RF 예측 840~867) 양 끝에서 선형보다 더 틀렸습니다. 다만 선형도 짧은 구간에서 +17.5% 과대예측하며, 각 구간 12셀이라 표본이 작습니다.
- **Linear + Gradient Boosting 평균 앙상블:** CV MAPE 7.5%로 선형 7.3%를 넘지 못하고 외삽도 더 나빠(20.4% / 16.1%) 채택하지 않았습니다.
- **타깃 log10 변환:** 전체 MAPE 7.7%로 원래 단위 7.3%보다 나빠 채택하지 않았습니다(짧은 구간만 소폭 개선).
- **보조 Feature 추가:** 선형 계열에서 오히려 나빠졌습니다(7.3% → 7.6~7.7%).

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

**Gap(Train–Valid) +4.03%p는 과적합으로 단정하지 않습니다.** Valid 11셀 중 한 정책(`3_6C-80PER_3_6C`, 수명 약 1,180)의 평균 APE가 19.2%로 가장 크고 나머지 네 정책은 7.4~11.0%입니다. 이 정책을 빼면 나머지 8셀 평균은 약 8.8%로 CV(7.60%)에 가깝습니다(정책을 제외하고 다시 학습한 것은 아니므로 가설입니다).

### 논문 비교용 참고: 분류

과제는 회귀/분류 중 하나만 택하도록 되어 있어 본 분석은 회귀입니다. 아래는 원논문의 분류 성능(1−Accuracy 4.9%)과 견주어 보기 위해 저장된 회귀 예측에 `cycle_life >= 550`(장수명)을 적용한 **참고 결과**이며, 모델 선택과 위 성능 표에는 쓰지 않았습니다(`02_DAY2_Modeling.ipynb` 13.2).

| 구분 | 1−Accuracy | 비고 |
|---|---:|---|
| Test (Batch 2) | 59.0% | 단수명 30셀 중 23셀을 장수명으로 오분류, 장수명 오분류 0건 |
| Target (원논문) | 4.9% | Gap +54.1%p |

- 입력이 달라 같은 조건의 비교가 아닙니다. 원논문은 초기 **5사이클**로 직접 분류했지만 여기서는 초기 100사이클 회귀 예측을 이진화했습니다.
- 전부 "단수명"으로 예측하는 기준선의 오차는 23.1%로, 이 값보다도 나쁩니다. Batch 1에 550 미만 셀이 1개뿐이라 경계를 학습하지 못한 것으로, 회귀에서 본 학습 수명 범위 문제와 같은 현상입니다.
- Batch 1 Valid 11셀은 전부 장수명이라 분류 성능을 평가할 수 없습니다.

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
| 선형모델이 트리 모델보다 외삽에 유리할 것이다 | 선형회귀의 CV 평균과 변동성이 더 낮았고, 별도 외삽 검증에서도 트리는 양 끝에서 선형보다 더 틀림(위 외삽 검증) | 적중: 트리 모델은 외삽과 소표본에 불리했음. 단, 선형도 단수명 영역을 과대예측해 Batch 2 성능 저하를 막지는 못함 |
| Batch 2 단수명 영역에서 외부 성능이 저하될 것이다 | Test MAPE 31.53%, Valid-Test Gap +19.90%p | 예상 위험 발생: 76.9%가 Batch 1 최소 수명 아래였음 |
| 원논문 MAPE 9.1%에 가까운 성능을 목표로 한다 | Target-Test Gap +22.43%p | 미달: 배치 이동, 외삽, 소표본, 재현 조건 차이 |

## 오류 분석

### 가장 크게 틀린 셀의 공통점
- Batch 2 라벨 셀 39개 중 30개(76.9%)가 Batch 1 최소 수명보다 짧았고, 30개 모두 실제보다 길게 예측됐습니다. 단수명 셀의 평균 APE는 36.73%, 나머지 셀은 14.20%였습니다.
- APE 상위 셀은 `3.6C(9%)-5C` 2셀(71.6%, 68.8%), `5.2C(50%)-4.25C` 3셀(67.3%, 52.0%, 45.9%), `5.2C(58%)-4C`(47.2%)입니다. 모두 실제 수명이 393~452사이클로 Batch 1 최소(534)보다 짧고 예측은 약 640~750사이클로 45~72% 과대예측입니다.
- 공통점은 ① 학습 수명 범위 밖의 단수명 셀이고, ② 충전 정책이 Batch 1에 없는 조건이라는 점입니다. 단수명 30셀의 정책 9종 중 Batch 1에도 있는 것은 2종(`4.8C(80%)-4.8C`, `6C(60%)-3C`)뿐이며 위 상위 오차 정책은 모두 Batch 1에 없습니다.
- 반대로 오차가 가장 작은 셀 3개(2.0%, 3.3%, 4.4%)는 모두 `NEWSTRUCTURE` 장수명 셀입니다.
- Batch 1–2 Target KS 통계량은 0.769, `deltaQ_logvar` KS 통계량은 0.591로 분포 이동이 컸습니다.

### 원인 가설과 확인한 것
- 같은 `deltaQ_logvar` 구간(−3.71~−3.09)에서 Batch 1 14셀의 실제 수명 평균은 659, Batch 2 단수명 30셀은 453으로 약 206사이클 짧습니다. 모델은 `deltaQ_logvar` 값만 보므로 이 차이가 그대로 과대예측이 됩니다.
- 단수명 셀의 평균 온도·IR은 대부분 Batch 1 범위 안(범위 밖 셀은 온도 2개, IR 3개)이라 이 입력 값의 이상은 원인이 아니었습니다. 과대예측과 같이 움직이는 다른 변수도 정책 단위(n=9)에서 약하고 일관되지 않았습니다(`02_DAY2_Modeling.ipynb` 13.1).
- **가설(확인되지 않음):** Batch 1에 없는 충전 조건이 수명에 미치는 영향이 `deltaQ_logvar`에 담기지 않았을 가능성이 있습니다. `newstructure`의 정확한 의미도 확인하지 못해 제조·측정 조건 차이를 배제하거나 단정할 수 없습니다. 단수명 셀이 정책 9종뿐이라 통계적으로도 약합니다.

### 개선 방향
- 단수명·다양한 충전 조건의 학습 셀 확충, 제조 배치가 다른 데이터로의 외부 검증, 예측 불확실성 제공.
- Batch 2를 이미 확인했으므로 이 개선안으로 현재 Test에 다시 맞추지 않고 후속 실험으로 분리합니다. 짧은 수명 셀을 학습에 포함하는 설계를 시도한다면 Batch 2 점수는 낙관적이므로 별도 표로 표시해야 합니다.

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
- 원논문의 9.1%·4.9%는 과제 안내에 제시된 값을 Target으로 사용했으며, 논문 본문 Table의 모델별 수치와 지표 정의·테스트셋 구성은 직접 확인하지 못했습니다.
- 원논문과 데이터 정정본, 전처리, Feature 계산, 분할 방식과 평가 범위가 완전히 같다고 보장할 수 없습니다.
- 상관계수와 모델 계수는 예측적 연관성이며 전기화학적 인과를 직접 증명하지 않습니다.
- 실제 배포 전에는 제조 배치·온도·운영조건이 다른 데이터로 외부 검증하고, 입력 분포 이동과 모델 성능을 지속적으로 모니터링해야 합니다.

## 참고문헌

- Severson, K. A. et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. *Nature Energy*, 4, 383–391.

## 팀 구성

- 문지후·정승우: 각자 독립적인 EDA·모델링 실험 수행, 결과 비교 및 최종 보고서 공동 작성
