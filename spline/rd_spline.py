# created by Steffen Urban, 11/2022

import theseus as th
import torch
from spline.spline_common import computeBaseCoefficients
from spline.spline_common import computeBlendingMatrix
from spline.spline_common import computeBaseCoefficientsWithTime
from spline.time_util import calc_times, S_TO_NS


class RDSpline:
    def __init__(self, start_time_ns, end_time_ns, dt_ns, dim=3, N=4, device="cpu"):
        self.dt_ns = dt_ns
        self.start_time_ns = start_time_ns
        self.end_time_ns = end_time_ns
        self.device = device
        # order of spline
        self.N = N
        # degree of spline
        self.DEG = N - 1
        # dimension of euclidean vector space
        self.DIM = dim

        self.blend_matrix = th.Variable(
            tensor=torch.tensor(computeBlendingMatrix(self.N, False)).to(self.device), 
            name="blend_matrix")
        self.base_coeffs = th.Variable(
            tensor=torch.tensor(computeBaseCoefficients(self.N)).to(self.device), 
            name="base_coeffs")

        # vector of knots. should be th.Vector
        self.knots = []

        self.update_inv_dt()

    def update_inv_dt(self):
        pow_inv_dt = torch.zeros(self.N).to(self.device)

        pow_inv_dt[0] = 1.0
        pow_inv_dt[1] = S_TO_NS / self.dt_ns

        for i in range(2, self.N):
            pow_inv_dt[i] = pow_inv_dt[i - 1] * pow_inv_dt[1]
        self.pow_inv_dt = th.Vector(tensor=pow_inv_dt, name="pow_inv_dt")

    def evaluate(self, time_ns, derivative=0):

        u, s, suc = calc_times(time_ns, 
            self.start_time_ns, self.dt_ns, 
            len(self.knots), self.N)  
        res = torch.zeros(1, self.DIM).float().to(self.device)

        if not suc:
            print("WRONG TIME")
            return res
            
        p = computeBaseCoefficientsWithTime(self.N, self.base_coeffs, derivative, u)
        coeff = self.pow_inv_dt.tensor[:,derivative] * (self.blend_matrix.tensor @ p)

        for i in range(self.N):
            res += coeff[i] * self.knots[s + i].tensor

        return res

    def velocity(self, time_ns):
        return self.evaluate(time_ns, 1)

    def acceleration(self, time_ns):
        return self.evaluate(time_ns, 2)

    def genRandomTrajectory(self,):
        num_knots = int((self.end_time_ns - self.start_time_ns) / self.dt_ns) + self.N
        for i in range(num_knots):
            self.knots.append(
                th.Vector(tensor=torch.randn(1,self.DIM), 
                    name="r"+str(self.DIM)+"_knot_"+str(i)))

        

# test
# r3_spline = RDSpline(0, 5*S_TO_NS, 0.1*S_TO_NS, 3, 4)
# r3_spline.genRandomTrajectory()

# print(r3_spline.evaluate(0*S_TO_NS))
# print(r3_spline.evaluate(1*S_TO_NS))
# print(r3_spline.evaluate(2*S_TO_NS))

# r3_spline = RDSpline(0, 5*S_TO_NS, 0.1*S_TO_NS, 2, 5)
# r3_spline.genRandomTrajectory()

# print(r3_spline.evaluate(0*S_TO_NS))
# print(r3_spline.evaluate(1*S_TO_NS))
# print(r3_spline.evaluate(2*S_TO_NS))
# print(r3_spline.velocity(2*S_TO_NS))
# print(r3_spline.acceleration(2*S_TO_NS))

# print(r3_spline.evaluate(6*S_TO_NS))

