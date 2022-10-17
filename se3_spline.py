
import theseus
import torch
from spline_common import computeBaseCoefficients
from spline_common import computeBlendingMatrix
from time_util import calc_times

class R3Spline:
    def __init__(self, N, dt_ns):
        self.dt_ns = dt_ns
        # order of spline
        self.N = N
        # degree of spline
        self.DEG = N - 1
        self.blend_matrix = torch.tensor(computeBlendingMatrix(5, False))
        self.base_coeffs = torch.tensor(computeBaseCoefficients)

        self.start_time_ns = 0

        self.nr_knots = 1


    def evaluate(self, time_ns):

        u, s, suc = calc_times(time_ns, 
            self.start_time_ns, self.dt_ns, 
            self.nr_knots, self.N)