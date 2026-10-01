"""Batch .mat(HDF5) -> 작은 캐시 파일. 원본(수 GB)을 매번 읽지 않기 위함.
사용: .venv/bin/python src/preprocess.py 1   (1, 2, 3)
결과: data/cache/batch{N}_summary.csv, batch{N}_dq.npz
"""
import sys, h5py, numpy as np, pandas as pd
from pathlib import Path

FILES = {1: '2017-05-12', 2: '2018-02-20', 3: '2018-04-12'}
ROOT = Path(__file__).resolve().parent.parent / 'data'
S_KEYS = ['IR', 'QCharge', 'QDischarge', 'Tavg', 'Tmax', 'Tmin', 'chargetime']


def text(f, ref):
    return ''.join(chr(c) for c in f[ref][:].flatten())


def run(n):
    f = h5py.File(ROOT / f'{FILES[n]}_batchdata_updated_struct_errorcorrect.mat', 'r')
    b = f['batch']
    rows, dq10, dq100, early_qd = [], [], [], []
    for i in range(b['summary'].shape[0]):
        s = f[b['summary'][i, 0]]
        d = {k: s[k][0] for k in S_KEYS}
        life = int(f[b['cycle_life'][i, 0]][0, 0])
        pol = text(f, b['policy_readable'][i, 0])
        for c in range(len(d['QDischarge'])):
            rows.append(dict(batch=n, cell=i, cycle=c + 1, cycle_life=life, policy=pol,
                             **{k: d[k][c] for k in S_KEYS}))
        cy = f[b['cycles'][i, 0]]
        # ΔQ(V) 용: 사이클 10, 100의 전압축 보간 방전용량 (1000포인트)
        # ponytail: 100사이클 미만 셀은 NaN 처리
        n_cy = cy['Qdlin'].shape[0]
        g = lambda k: f[cy['Qdlin'][k, 0]][:].flatten() if k < n_cy else np.full(1000, np.nan)
        dq10.append(g(9)); dq100.append(g(99))
    out = ROOT / 'cache'; out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out / f'batch{n}_summary.csv', index=False)
    np.savez(out / f'batch{n}_dq.npz', q10=np.array(dq10), q100=np.array(dq100))
    print(f'batch{n}: {len(dq10)} cells -> {out}')


if __name__ == '__main__':
    run(int(sys.argv[1]))
