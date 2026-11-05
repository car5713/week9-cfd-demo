"""Generate CFD reference fields for the extrapolation comparison.

The solver functions are the Week 9 course solver, unchanged (workflow inspired by
Barba & Forsyth, CFD Python Step 11, https://github.com/barbagroup/CFDPython).
Only max_steps is larger, because high-Re cases take longer to reach steady state.
These fields check the network's predictions. They are never used to train it.

Usage: python cfd_reference.py [output.json]
"""
import base64, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu
from numba import njit


def pressure_factor(n):
    h=1.0/n
    A=lil_matrix((n*n,n*n))
    for j in range(n):
        for i in range(n):
            k=j*n+i
            for jj,ii in [(j-1,i),(j+1,i),(j,i-1),(j,i+1)]:
                if 0<=jj<n and 0<=ii<n:
                    A[k,jj*n+ii]=1/h**2
                    A[k,k]-=1/h**2
    A[0,:]=0; A[0,0]=1
    return splu(A.tocsc())

@njit
def tentative_velocity(u,v,nu,dt,h):
    n=u.shape[0]; us=u.copy(); vs=v.copy()
    for j in range(n):
        for i in range(1,n):
            uc=u[j,i]
            vc=.25*(v[j,i-1]+v[j,i]+v[j+1,i-1]+v[j+1,i])
            below=u[j-1,i] if j>0 else -uc
            above=u[j+1,i] if j<n-1 else 2.0-uc
            dx=(uc-u[j,i-1])/h if uc>=0 else (u[j,i+1]-uc)/h
            dy=(uc-below)/h if vc>=0 else (above-uc)/h
            lap=(u[j,i-1]+u[j,i+1]+below+above-4*uc)/(h*h)
            us[j,i]=uc+dt*(-uc*dx-vc*dy+nu*lap)
    for j in range(1,n):
        for i in range(n):
            vc=v[j,i]
            uc=.25*(u[j-1,i]+u[j-1,i+1]+u[j,i]+u[j,i+1])
            left=v[j,i-1] if i>0 else -vc
            right=v[j,i+1] if i<n-1 else -vc
            dx=(vc-left)/h if uc>=0 else (right-vc)/h
            dy=(vc-v[j-1,i])/h if vc>=0 else (v[j+1,i]-vc)/h
            lap=(left+right+v[j-1,i]+v[j+1,i]-4*vc)/(h*h)
            vs[j,i]=vc+dt*(-uc*dx-vc*dy+nu*lap)
    return us,vs

def solve_cavity(reynolds,n=50,max_steps=20000,steady_tol=2e-5,factor=None):
    h=1/n; nu=1/reynolds; dt=min(.2*h,.2*h*h/nu)
    factor=pressure_factor(n) if factor is None else factor
    u=np.zeros((n,n+1)); v=np.zeros((n+1,n)); p=np.zeros((n,n))
    change=np.inf
    for step in range(max_steps):
        us,vs=tentative_velocity(u,v,nu,dt,h)
        rhs=((us[:,1:]-us[:,:-1]+vs[1:,:]-vs[:-1,:])/h/dt).ravel()
        rhs-=rhs.mean(); rhs[0]=0
        p=factor.solve(rhs).reshape(n,n)
        us[:,1:-1]-=dt*(p[:,1:]-p[:,:-1])/h
        vs[1:-1,:]-=dt*(p[1:,:]-p[:-1,:])/h
        change=max(np.max(np.abs(us-u)),np.max(np.abs(vs-v)))/dt
        u,v=us,vs
        if not np.isfinite(change) or np.max(np.abs(u))>10:
            raise ValueError('CFD unstable: no field accepted.')
        if step>100 and change<steady_tol: break
    div=(u[:,1:]-u[:,:-1]+v[1:,:]-v[:-1,:])/h
    p-=p.mean()
    field=np.stack(((u[:,1:]+u[:,:-1])/2,(v[1:,:]+v[:-1,:])/2,p),axis=-1)
    return field,step+1,change,np.sqrt(np.mean(div**2))


# In-range cases only give the error chart a baseline; Re >= 300 are the comparison cases.
REYNOLDS = [20, 50, 100, 150, 200, 250] + list(range(300, 1001, 50))


def run_case(re):
    t0 = time.perf_counter()
    field, steps, residual, div = solve_cavity(float(re), max_steps=400000)
    if residual >= 2e-5 or div >= 1e-8:
        raise RuntimeError(f"Re={re} failed the course acceptance checks: {residual=}, {div=}")
    print(f"Re={re:5d} steps={steps:7d} residual={residual:.2e} div={div:.2e} {time.perf_counter()-t0:.0f}s", flush=True)
    data = field.astype("<f4").tobytes()  # [y_cell, x_cell, variable], row order increasing y
    return {"re": re, "steps": steps, "residual": residual, "divergence": div,
            "fields_b64": base64.b64encode(data).decode("ascii")}


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "cfd_reference.json"
    with ProcessPoolExecutor(max_workers=8) as pool:
        cases = list(pool.map(run_case, REYNOLDS))
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"solver": "week9 course MAC solver, n=50", "n": 50, "layout": "float32le [y,x,(u,v,p)]",
                   "cases": cases}, f, separators=(",", ":"))
    print("wrote", out)
