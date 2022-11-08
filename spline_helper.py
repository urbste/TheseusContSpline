import torch
import theseus as th
from spline_common import computeBaseCoefficients, computeBaseCoefficientsWithTimeVec
from spline_common import computeBlendingMatrix
from spline_common import computeBaseCoefficientsWithTime

import time

class SplineHelper:

    def __init__(self, N, device="cpu"):
        self.N = N
        self.DEG =self.N - 1
        self.device = device
        self.blending_matrix = torch.tensor(
            computeBlendingMatrix(N, cumulative=False)).to(self.device)
        self.cumulative_blending_matrix = torch.tensor(
            computeBlendingMatrix(N, cumulative=True)).to(self.device)
        self.base_coefficients = torch.tensor(
            computeBaseCoefficients(N)).to(self.device)
    
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

        rot_vel, rot_accel = th.Vector(3), th.Vector(3)

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

    def evaluate_euclidean_vec(self,
        rd_knots,
        u,
        inv_dt,
        derivatives=0,
        num_meas=0):

        p = computeBaseCoefficientsWithTimeVec(
            self.N, self.base_coefficients, derivatives, u, num_meas)
        pow_inv_dt = inv_dt ** derivatives

        coeff = pow_inv_dt * (self.blending_matrix @ p)

        scaled = coeff.unsqueeze(-1) * rd_knots
        res = torch.sum(scaled,0)
        return res

    def evaluate_lie_vec(self,
        so3_knots,
        u,
        inv_dt, 
        derivatives,
        num_meas=0):

        p = computeBaseCoefficientsWithTimeVec(
            self.N, self.base_coefficients, derivative=0, u=u,num_pts=num_meas)
        coeff = self.cumulative_blending_matrix @ p
        if derivatives >= 1:
            p = computeBaseCoefficientsWithTimeVec(
                self.N, self.base_coefficients, derivative=1, u=u,num_pts=num_meas)
            dcoeff = inv_dt * (self.cumulative_blending_matrix @ p)
        if derivatives >= 2:
            p = computeBaseCoefficientsWithTimeVec(
                self.N, self.base_coefficients, derivative=2, u=u,num_pts=num_meas)
            ddcoeff = inv_dt * inv_dt * (self.cumulative_blending_matrix @ p)

        transform_out = th.SO3(tensor=so3_knots[0])

        rot_vel, rot_accel = torch.zeros(num_meas, 3, 1).to(self.device), torch.zeros(num_meas, 3, 1).to(self.device)

        for i in range(0, self.DEG):
            p0 = th.SO3(tensor=so3_knots[i])
            p1 = th.SO3(tensor=so3_knots[i + 1])
            r01 = p0.inverse().compose(p1)
            delta = r01.log_map()
            kdelta = coeff[i+1].unsqueeze(1) * delta
            exp_kdelta = th.SO3.exp_map(tangent_vector=kdelta)

            transform_out = transform_out.compose(exp_kdelta)

            if derivatives >= 1:
                Adj = exp_kdelta.inverse().adjoint()
                rot_vel = Adj @ rot_vel
                rot_vel_current = dcoeff[i + 1].unsqueeze(1) * delta 
                rot_vel = rot_vel + rot_vel_current.unsqueeze(-1)
            if derivatives >= 2:
                rot_accel = Adj @ rot_accel
                accel_lie_bracket = torch.cross(rot_vel.tensor, rot_vel_current.tensor)
                rot_accel = rot_accel + (ddcoeff[i + 1].unsqueeze(1) * delta + accel_lie_bracket)
        return transform_out, rot_vel, rot_accel