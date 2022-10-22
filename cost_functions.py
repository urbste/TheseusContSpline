

from typing import List, Optional, Tuple
import torch
import theseus as th

from spline_helper import SplineHelper

spline_helper = SplineHelper(N=4)

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
        knots, u, inv_dt, derivative=1, num_meas=len(u))[1]

    return rot_vel.squeeze(-1) - meas

def r3_accl_error(optim_vars, aux_vars):

    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3)
    rot_vel = spline_helper.evaluate_euclidean_vec(
        knots, u, inv_dt, derivative=2, num_meas=len(u))[1]

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
        knots, u, inv_dt, derivative=1, num_meas=len(u))[1]

    return rot_vel.squeeze(-1) - meas

def so3_accl_error(optim_vars, aux_vars):
    knots, meas, u, inv_dt, sdim = extract_spline_vars(
        optim_vars, aux_vars)
    knots = knots.reshape(sdim, len(u), 3, 3)
    rot_accl = spline_helper.evaluate_lie_vec(
        knots, u, inv_dt, derivative=2, num_meas=len(u))[1]

    return rot_accl.squeeze(-1) - meas

