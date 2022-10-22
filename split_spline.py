
import theseus as th
import torch
from spline_common import computeBaseCoefficients, computeBaseCoefficientsWithTime
from spline_common import computeBlendingMatrix
from spline_common import computeBaseCoefficientsWithTime
from time_util import calc_times, S_TO_NS

class SplitSpline3:
    def __init__(self, start_time_ns, end_time_ns, dt_ns, N=4):
        self.dt_ns = dt_ns
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns

        # order of spline
        self.N = N
        # degree of spline
        self.DEG = N - 1
        # dimension of euclidean vector space
        self.DIM = 3

        self.blend_matrix = th.Variable(
            tensor=torch.tensor(computeBlendingMatrix(self.N, True)), 
            name="blend_matrix")
        self.base_coeffs = th.Variable(
            tensor=torch.tensor(computeBaseCoefficients(self.N)), 
            name="base_coeffs")


        self.update_inv_dt()

    def update_inv_dt(self):
        pow_inv_dt = torch.zeros(self.N)

        pow_inv_dt[0] = 1.0
        pow_inv_dt[1] = S_TO_NS / self.dt_ns

        for i in range(2, self.N):
            pow_inv_dt[i] = pow_inv_dt[i - 1] * pow_inv_dt[1]
        self.pow_inv_dt = th.Vector(tensor=pow_inv_dt, name="pow_inv_dt")

    def evaluate(self, time_ns):

        u, s, suc = calc_times(time_ns, 
            self.start_time_ns, self.dt_ns, 
            len(self.knots), self.N)  

        res = self.knots[s]

        if not suc:
            print("WRONG TIME")
            return res

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 0, u)
        coeff = self.blend_matrix.tensor @ p

        for i in range(self.DEG):
            p0 = self.knots[s + i]
            p1 = self.knots[s + i + 1]
            r01 = p0.inverse().compose(p1)
            delta = r01.log_map()
            kdelta = delta * coeff[i+1]
            res.compose(th.SO3.exp_map(kdelta))
        return res

    def velocityBody(self, time_ns):
        u, s, suc = calc_times(time_ns, 
            self.start_time_ns, self.dt_ns, 
            len(self.knots), self.N)  
        res = torch.zeros(1, self.DIM).float()

        if not suc:
            print("WRONG TIME")
            return res

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 0, u)
        coeff = self.blend_matrix.tensor @ p

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 1, u)
        dcoeff = self.pow_inv_dt.tensor[:,1] * self.blend_matrix.tensor @ p 

        rot_vel = torch.zeros(3).float()
        
        for i in range(self.DEG):
            p0 = self.knots[s + i]
            p1 = self.knots[s + i + 1]
            r01 = p0.inverse().compose(p1)
            delta = r01.log_map()
            rot_vel = th.SO3.exp_map(-delta * coeff[i + 1]).to_matrix().squeeze(0) @ rot_vel
            rot_vel += delta.squeeze(0) * dcoeff[i + 1]

        return rot_vel

    def accelerationBody(self, time_ns):
        u, s, suc = calc_times(time_ns, 
            self.start_time_ns, self.dt_ns, 
            len(self.knots), self.N)  
        res = torch.zeros(self.DIM, 1).float()

        if not suc:
            print("WRONG TIME")
            return res

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 0, u)
        coeff = self.blend_matrix.tensor @ p

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 1, u)
        dcoeff = self.pow_inv_dt.tensor[:,1] * self.blend_matrix.tensor @ p 

        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, 2, u)
        ddcoeff = self.pow_inv_dt.tensor[:,2] * self.blend_matrix.tensor @ p 
        
        rot_vel = torch.zeros(3).float()

        rot_accel = torch.zeros(3).float()

        for i in range(self.DEG):
            p0 = self.knots[s + i]
            p1 = self.knots[s + i + 1]
            r01 = p0.inverse().compose(p1)
            delta = r01.log_map()

            rot = th.SO3.exp_map(-delta * coeff[i + 1]).to_matrix().squeeze(0)

            rot_vel = rot @  rot_vel
            vel_current = delta.squeeze(0) * dcoeff[i + 1]
            rot_vel += vel_current

            rot_accel = rot @ rot_accel
            rot_accel += ddcoeff[i + 1] * delta.squeeze(0) + torch.cross(rot_vel, vel_current)

        return rot_accel


    def genRandomTrajectory(self,):
        num_knots = int((self.end_time_ns - self.start_time_ns) / self.dt_ns) + self.N
        for i in range(num_knots):
            self.knots.append(
                th.SO3(name="so3_knot_"+str(i)).randn(1))

        
