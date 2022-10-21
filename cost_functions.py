

from typing import List, Optional, Tuple
import torch
import theseus as th

from spline_helper import SplineHelper

spline_helper = SplineHelper(N=4)

def spline_position_error(optim_vars, aux_vars):

    pos_measurement = aux_vars[0].tensor
    u = aux_vars[1].tensor
    inv_dt = aux_vars[2].tensor
    knots_list = [knot.tensor for knot in optim_vars]
    knots = torch.cat(knots_list,0)
    spline_dim = len(knots_list)
    vector_dim = 2
    knots = knots.reshape(spline_dim, len(u), vector_dim)
    spline_position = spline_helper.evaluate_euclidean_vec(knots, u, inv_dt, 0, len(u))
    return spline_position - pos_measurement

def spline_rot_error(optim_vars, aux_vars):

    R_w_s_measured = aux_vars[0].tensor
    u = aux_vars[1].tensor
    inv_dt = aux_vars[2].tensor
    knots_list = [knot.tensor for knot in optim_vars]
    knots = torch.cat(knots_list,0)
    spline_dim = len(knots_list)
    knots = knots.reshape(spline_dim, len(u), 3, 3)
    R_w_s = spline_helper.evaluate_lie_vec(knots, u, inv_dt, 0, len(u))[0]
    return th.SO3(tensor=R_w_s_measured).compose(R_w_s.inverse()).log_map() # th.Between(R_w_s, R_w_s_measured, )

# class PositionError2(th.CostFunction):
#     def __init__(
#         self,
#         r2_spline_knots: List[th.Vector],
#         pos_measurement: th.Point3,
#         u: float, 
#         inv_dt: float,
#         weight: Optional[th.CostWeight] = None,
#         name: Optional[str] = None,
#     ):
#         if weight is None:
#             weight = th.ScaleCostWeight(torch.tensor(1.0).to(dtype=pos_measurement.dtype))
#         super().__init__(
#             cost_weight=weight,
#             name=name,
#         )

#         self.r2_spline_knots = r2_spline_knots
#         self.pos_measurement = pos_measurement
#         self.u = u
#         self.inv_dt = inv_dt
#         self.spline_helper = spline_helper
#         self.register_optim_vars(["r3_spline_knots"])
#         self.register_aux_vars(["pos_measurement", "u", "inv_dt"])

#     def error(self) -> torch.Tensor:
        
#         spline_position = spline_helper.evaluate_euclidean(
#             self.r2_spline_knots, self.u, self.inv_dt, 0)
#         err = spline_position - self.pos_measurement.tensor
#         return err

#     def dim(self) -> int:
#         return 2

#     def to(self, *args, **kwargs):
#         super().to(*args, **kwargs)

#     def _copy_impl(self, new_name: Optional[str] = None) -> "PositionError2":
#         return PositionError2(
#             self.r2_spline_knots.copy(),
#             self.pos_measurement.copy(),
#             self.u.copy(),
#             self.inv_dt.copy(),
#             self.spline_helper.copy(),
#             weight=self.weight.copy(),
#             name=new_name,
#         )

