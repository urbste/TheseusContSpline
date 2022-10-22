

from typing import List, Optional, Tuple
import torch
import theseus as th

from spline_helper import SplineHelper

spline_helper = SplineHelper(N=4)

def extract_se3_spline_vars(optim_vars, aux_vars):
    meas = aux_vars[0].tensor
    u_r3 = aux_vars[1].tensor
    u_so3 = aux_vars[2].tensor
    inv_dt_r3 = aux_vars[3].tensor
    inv_dt_so3 = aux_vars[4].tensor
    knots_list = [knot.tensor for knot in optim_vars]
    knots = torch.cat(knots_list,0)
    spline_dim = len(knots_list)

    return meas, [u_r3, u_so3], [inv_dt_r3, inv_dt_so3], knots, spline_dim

def extract_spline_vars(optim_vars, aux_vars):
    measurements = aux_vars[0].tensor
    u = aux_vars[1].tensor
    inv_dt = aux_vars[2].tensor
    knots_list = [knot.tensor for knot in optim_vars]
    knots = torch.cat(knots_list,0)
    spline_dim = len(knots_list)

    return knots, measurements, u, inv_dt, spline_dim

def r3_error(optim_vars, aux_vars):
    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3)
    spline_position = spline_helper.evaluate_euclidean_vec(
        knots, u, inv_dt, 0, len(u))
    return spline_position - meas

def r3_vel_error(optim_vars, aux_vars):
    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3)
    rot_vel = spline_helper.evaluate_euclidean_vec(
        knots, u, inv_dt, derivatives=1, num_meas=len(u))

    return rot_vel.squeeze(-1) - meas

def r3_accl_error(optim_vars, aux_vars):

    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3)
    rot_vel = spline_helper.evaluate_euclidean_vec(
        knots, u, inv_dt, derivatives=2, num_meas=len(u))

    return rot_vel.squeeze(-1) - meas

def so3_error(optim_vars, aux_vars):

    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3, 3)
    R_w_s = spline_helper.evaluate_lie_vec(
        knots, u, inv_dt, 0, len(u))[0]
    return th.SO3(tensor=meas).compose(R_w_s.inverse()).log_map() 

def so3_vel_error(optim_vars, aux_vars):
    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3, 3)
    rot_vel = spline_helper.evaluate_lie_vec(
        knots, u, inv_dt, derivatives=1, num_meas=len(u))[1]

    return rot_vel.squeeze(-1) - meas

def so3_accl_error(optim_vars, aux_vars):
    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3, 3)
    rot_accl = spline_helper.evaluate_lie_vec(
        knots, u, inv_dt, derivatives=2, num_meas=len(u))[2]

    return rot_accl.squeeze(-1) - meas

def accelerometer_error(optim_vars, aux_vars):
    meas, us, inv_dts, knots, sdim = extract_se3_spline_vars(
        optim_vars, aux_vars)
    knots_se3 = knots.reshape(sdim, len(us[0]), 3, 4)

    u_r3, u_so3 = us
    i_dt_r3, i_dt_so3 = inv_dts
    R_w_i = spline_helper.evaluate_lie_vec(
        knots_se3[:,:,:3,:3], u_so3, i_dt_so3, derivatives=0, num_meas=len(u_so3))[0]

    accl_w = spline_helper.evaluate_euclidean_vec(
        knots_se3[:,:,:3,3], u_r3, i_dt_r3, derivatives=2, num_meas=len(u_r3))

    # add bias and so on
    spline_accl = R_w_i.inverse().to_matrix() @ (accl_w + torch.tensor([0.,0.,9.81])).unsqueeze(-1)
    return spline_accl - meas

