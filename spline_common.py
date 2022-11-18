import numpy as np
import math
import torch
# def C_n_k(n, k):
#     if k > n:
#         return 0   
#     r = 1
#     d = 1
#     n_ = n
#     while d <= k:
#         r = r * n
#         n -= 1
#         r /= d
#         d += 1
#     return r

def computeBlendingMatrix(N, cumulative):
    '''
    N : int
        order of spline
    cumulative : boolean
        if the spline should be cumulative
    '''
    blend_mat = np.zeros((N,N), dtype=np.float32)
    for i in range(N):
        for j in range(N):
            sume = 0
            for s in range(j,N):
                sume += (       -1.0)**(s - j) * math.comb(N, s-j) * \
                        (N - s - 1.0)**(N - 1.0 - i)

            blend_mat[j, i] = math.comb(N - 1, N - 1 - i) * sume

    if cumulative:
        for i in range(N):
            for j in range(i, N):
                blend_mat[i,:] += blend_mat[j,:]
    
    factorial = 1.0
    for i in range(2, N):
        factorial *= i
    
    return blend_mat / factorial

def computeBaseCoefficients(N):
    base_coeffs = np.zeros((N,N), dtype=np.float32)
    base_coeffs[0,:] = 1.0
    DEG = N - 1
    order = DEG
    for n in range(1, N):
        for i in range(DEG - order, N):
            base_coeffs[n, i] = (order-DEG+i) * base_coeffs[n-1, i]
        order -= 1

    return base_coeffs

def computeBaseCoefficientsWithTime(N, base_coeffs, derivative, u):
    res = torch.zeros(N,1).float().to(base_coeffs.device)

    if derivative < N:
        res[derivative] = base_coeffs[derivative, derivative]
        _t = u
        for j in range(derivative+1, N):
            res[j] = base_coeffs[derivative, j] * _t
            _t = _t * u

    return res

def computeBaseCoefficientsWithTimeVec(N, base_coeffs, derivative, u, num_pts):

    results = []
    if derivative < N:
        for _ in range(0,derivative):
            results.append(torch.tensor([0.]).repeat(num_pts))
        results.append(base_coeffs[derivative, derivative].repeat(num_pts))
        _t = u
        for j in range(derivative+1, N):
            results.append(base_coeffs[derivative, j] * _t)
            _t = _t * u

    return torch.stack(results,0)
