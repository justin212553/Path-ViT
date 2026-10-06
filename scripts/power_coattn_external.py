"""
2026-10-04: PAAD external(CPTAC) co-attention 효과(M4-SA -> M4 paired dC)의 검정력 vs external 표본 크기.
실제 external 앙상블 예측에서 N명 복원추출 400회 -> dC 표준오차 -> 양측 a=0.05 검정력(정규근사).
결과는 paper/TODO_reviewer_feedback.md 0절 참고. 사용법: python scripts/power_coattn_external.py
"""
import sys, numpy as np
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from analyze_foldsafe_batch import load, PAAD
from utils.metrics import compute_survival_metrics

def cidx(r,t,e):
    # Harrell: comparable (i,j) if e_i and t_i<t_j; concordant if r_i>r_j
    d = (t[:,None] < t[None,:]) & (e[:,None]==1)
    rr = r[:,None]-r[None,:]
    return ((rr>0)&d).sum()+0.5*((rr==0)&d).sum(), d.sum()

def get(name):
    ids,r,t,e = load("PAAD","external",PAAD[name]); return dict(zip(ids,zip(r,t,e)))
rng = np.random.RandomState(0)
for a,b in [("M4-SA","M4"),("M4-SA-FIXC","M4-FIXC")]:
    A,B = get(a),get(b); ids = sorted(set(A)&set(B))
    ra=np.array([A[i][0] for i in ids]); rb=np.array([B[i][0] for i in ids])
    t=np.array([A[i][1] for i in ids],float); e=np.array([A[i][2] for i in ids],int)
    na,da=cidx(ra,t,e); nb,_=cidx(rb,t,e); delta=(nb-na)/da
    chk = compute_survival_metrics(rb,t,e)["c_index"]-compute_survival_metrics(ra,t,e)["c_index"]
    print(f"\n{a} -> {b}: N={len(ids)} events={e.sum()} dC={delta:+.4f} (check {chk:+.4f})")
    for N in [136,200,300,400,500,700,1000]:
        ds=[]
        for _ in range(400):
            ix=rng.randint(0,len(ids),N)
            n1,d1=cidx(ra[ix],t[ix],e[ix]); n2,_=cidx(rb[ix],t[ix],e[ix]); ds.append((n2-n1)/d1)
        ds=np.array(ds); se=ds.std()
        from scipy.stats import norm
        pw=norm.cdf(abs(delta)/se-1.96)+norm.cdf(-abs(delta)/se-1.96)
        pw25=norm.cdf(0.025/se-1.96)
        print(f"  N={N:5d} (~{int(N*e.mean())} events) SE={se:.4f} power@obs={pw:.2f} power@0.025={pw25:.2f} SE*sqrt(N)={se*np.sqrt(N):.3f}")
