# created by Steffen Urban, 11/2022
import torch

class GyroscopeRes:
    def __init__(self, inv_dt_so3, spline_helper):
        # Todo add imu intrinsics?
        self.inv_dt_so3 = inv_dt_so3
        self.spline_helper = spline_helper

    def __call__(self, optim_vars, aux_vars):

        norm_times = aux_vars[0].tensor
        
        so3_knots_tensor_ = torch.stack([so3.tensor for so3 in optim_vars], dim=0)
        # evaluate first derivative of SO3 spline to get angular velocity
        rot_vel_w_i = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_, norm_times, 
            self.inv_dt_so3.tensor, derivatives=1)[1]
    
        return rot_vel_w_i.squeeze() - aux_vars[1].tensor


class AccelerometerRes:
    def __init__(self, inv_dt_so3, inv_dt_r3, spline_helper, gravity=9.81):
        # Todo add imu intrinsics?
        self.inv_dt_so3 = inv_dt_so3
        self.inv_dt_r3 = inv_dt_r3
        self.spline_helper = spline_helper
        self.gravity = gravity

    def __call__(self, optim_vars, aux_vars):

        norm_times = aux_vars[0].tensor
        u_so3 = norm_times[:,0]
        u_r3 = norm_times[:,1]
        so3_knots_ = optim_vars[: len(optim_vars) // 2]  # each (B, 3, 3)
        r3_knots_ = optim_vars[len(optim_vars) // 2 :]   # each (B, 3)
        so3_knots_tensor_ = torch.stack([so3.tensor for so3 in so3_knots_], dim=0)
        r3_knots_tensor_ = torch.stack([so3.tensor for so3 in r3_knots_], dim=0)

        # evaluate so3 spline to get imu to world rotation
        R_w_i = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_, u_so3, 
            self.inv_dt_so3.tensor, derivatives=0)[0]
        # evaluate r3 spline to get accleration in world
        accel_w = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_, u_r3, 
            self.inv_dt_so3.tensor, derivatives=2)

        # accel in body
        R_w_i_inv = R_w_i.inverse()
        accel_b = R_w_i_inv.tensor @ (accel_w + self.gravity).unsqueeze(-1) # - bias_accel

        return accel_b.squeeze() - aux_vars[1].tensor