# data/

원본 데이터(`*.mat`)와 전처리 캐시(`cache/`)는 용량이 커서 GitHub에 올리지 않습니다. 아래 순서로 직접 준비합니다.

## 1. 원본 받기

- Kaggle: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle (zip 약 5GB)
- 압축을 풀고 아래 3개 파일을 이 폴더(`data/`)에 둡니다.

| 파일 | Batch | 용도 |
| --- | --- | --- |
| `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | Batch 1 | 학습 |
| `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | Batch 2 | 최종 평가 |
| `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | Batch 3 | EDA 비교용 (선택) |

`2018-04-03_varcharge_...mat`은 다른 논문(Attia 2020)용이라 사용하지 않습니다.

## 2. 캐시 만들기

`.mat`(수 GB)를 매번 읽지 않도록, 필요한 값만 뽑아 `data/cache/`에 저장합니다. 프로젝트 루트에서 한 번만 실행합니다.

```bash
python src/preprocess.py 1
python src/preprocess.py 2
python src/preprocess.py 3
```

| 생성 파일 | 내용 |
| --- | --- |
| `cache/batch{N}_summary.csv` | 셀·사이클별 요약(QD, IR, 온도, 충전시간 등), `cycle_life`, 충전 정책 |
| `cache/batch{N}_dq.npz` | 사이클 10, 100의 전압축 보간 방전용량(`Qdlin`, 1000포인트) → ΔQ(V) 계산용 |

캐시는 언제든 위 명령으로 다시 만들 수 있습니다.

## 3. 사용

서브 노트북(`notebooks/01_DAY1_EDA_sub.ipynb`, `notebooks/02_hypothesis_H1_H2.ipynb`)은 `data/cache/`의 파일을 읽습니다. 메인 EDA(`01_DAY1_EDA.ipynb`)는 원본 `.mat`를 직접 읽고, 모델링(`02_DAY2_Modeling.ipynb`)은 `results/battery_eda/cell_features.csv`를 읽습니다.

```python
df = pd.read_csv('../data/cache/batch1_summary.csv')
z = np.load('../data/cache/batch1_dq.npz')   # z['q10'], z['q100'] : (셀 수, 1000)
```

## 참고

- `cycle_life`가 비어 있는(NaN) 셀은 EOL(80%)에 도달하지 못했거나 특수 실험 셀이며, 분석에서 제외합니다 (Batch 2: 8개, Batch 3: 2개).
- 전압축은 3.5V → 2.0V(1000포인트)입니다.
