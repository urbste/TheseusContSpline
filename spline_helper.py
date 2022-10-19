import torch
import theseus as th
from spline_common import computeBaseCoefficients
from spline_common import computeBlendingMatrix
from spline_common import computeBaseCoefficientsWithTime



class SplineHelper:

    def __init__(self, N):
        self.N = N
        self.DEG =self.N - 1
        self.blending_matrix = torch.tensor(
            computeBlendingMatrix(N, cumulative=False))
        self.cumulative_blending_matrix = torch.tensor(
            computeBlendingMatrix(N, cumulative=True))
        self.base_coefficients = torch.tensor(
            computeBaseCoefficients(N))
    
    def evaluate_lie(self,
        so3_knots,
        u,
        inv_dt, 
        derivative):

        p = computeBaseCoefficientsWithTime(
            self.N, self.base_coefficients, derivative=0, u=u)
        coeff = self.cumulative_blending_matrix @ p
        if derivative >= 1:
            p = computeBaseCoefficientsWithTime(
                self.N, self.base_coefficients, derivative=1, u=u)
            dcoeff = inv_dt * self.cumulative_blending_matrix @ p
        if derivative >= 2:
            p = computeBaseCoefficientsWithTime(
                self.N, self.base_coefficients, derivative=2, u=u)
            ddcoeff = inv_dt * inv_dt * self.cumulative_blending_matrix @ p

        transform_out = self.so3_knots[0]

        rot_vel, rot_accel = th.Vector(3), th.Vector3

        for i in range(0, self.DEG):
            p0 = th.SO3(so3_knots[i])
            p1 = th.SO3(so3_knots[i + 1])
            r01 = p0.inverse().compose(p1)
            delta = r01.log_map()
            kdelta = delta * coeff[i+1]
            exp_kdelta = th.SO3.exp_map(kdelta)
            transform_out.compose(exp_kdelta)

            if derivative >= 1:
                Adj = exp_kdelta.inverse().adjoint()
                rot_vel = Adj.to_matrix() * rot_vel
                rot_vel_current = delta * dcoeff[i + 1]
                rot_vel += rot_vel_current
            if derivative >= 2:
                rot_accel = Adj @ rot_accel
                accel_lie_bracket = torch.cross(rot_vel.tensor, rot_vel_current.tensor)
                rot_accel += ddcoeff[i + 1] * delta + accel_lie_bracket

        return transform_out, rot_vel, rot_accel

    def evaluate_euclidean(self,
        rd_knots,
        u,
        inv_dt,
        derivative=0):

        # get dimenstion of vector space
        _, DIM = rd_knots[0].shape

        res = torch.zeros(1, DIM).float()

        p = computeBaseCoefficientsWithTime(
            self.N, self.base_coefficients, derivative, u)
        pow_inv_dt = inv_dt ** derivative

        coeff = pow_inv_dt[:,derivative] * (self.blending_matrix @ p)

        for i in range(self.N):
            res += coeff[i] * rd_knots[i].tensor

        return res

