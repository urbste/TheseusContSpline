

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


def extract_rs_reproj_se3_spline_vars(optim_vars, aux_vars):
    aux_dict = {}
    aux_dict["meas_ref"] = aux_vars[0].tensor
    aux_dict["meas_obs"] = aux_vars[1].tensor
    # reference and observation times
    aux_dict["u_r3_ref"] = aux_vars[2].tensor
    aux_dict["u_so3_ref"] = aux_vars[3].tensor
    aux_dict["u_r3_obs"] = aux_vars[4].tensor
    aux_dict["u_so3_obs"] = aux_vars[5].tensor
    aux_dict["inv_dt_r3"] = aux_vars[6].tensor
    aux_dict["inv_dt_so3"] = aux_vars[7].tensor
    aux_dict["ref_bearings"] = aux_vars[8].tensor
    aux_dict["T_i_c"] = aux_vars[9].tensor
    aux_dict["line_delay"] = aux_vars[10].tensor
    aux_dict["cam_matrix"] = aux_vars[11].tensor
    aux_dict["inv_depth"] = aux_vars[12].tensor

    opt_dict = {}
    knots_list = [knot.tensor for knot in optim_vars]
    opt_dict["knots"] = torch.cat(knots_list,0)
    spline_dim = len(knots_list)
    #opt_dict["inv_depths"] = 
    #opt_dict["line_delay"] = 
    return aux_dict, opt_dict, spline_dim


def rs_reproj_error(optim_vars, aux_vars):


    # meas_ref = aux_vars[0].tensor
    # meas_obs = aux_vars[1].tensor
    # u_r3_ref = aux_vars[2].tensor
    # u_so3_ref = aux_vars[3].tensor
    # u_r3_obs = aux_vars[4].tensor
    # u_so3_obs = aux_vars[5].tensor
    # inv_dt_r3 = aux_vars[6].tensor
    # inv_dt_so3 = aux_vars[7].tensor
    # ref_bearings = aux_vars[8].tensor
    # T_c_i = aux_vars[9].tensor
    # line_delay = aux_vars[10].tensor
    # cam_matrix = aux_vars[11].tensor

    r3_knots = optim_vars[:len(optim_vars)//2]
    so3_knots = optim_vars[len(optim_vars)//2:]
    
    th.SE3()

    s = aux_vars[0].tensor
    u = aux_vars[1].tensor

    # (obs, N, )
    s_so3_ref = s[0,:,:,0].int()
    s_r3_ref = s[0,:,:,1].int()
    s_so3_obs = s[0,:,:,2].int()
    s_r3_obs = s[0,:,:,3].int()

    u_so3_ref = u[0,:,0]
    u_r3_ref = u[0,:,1]
    u_so3_obs = u[0,:,2]
    u_r3_obs = u[0,:,3]

    num_fs = len(aux_dict["u_r3_ref"])
    knots_se3 = opt_dict["knots"].reshape(sdim, num_fs, 3, 4)


    y_coord_ref_t_ns = aux_dict["meas_ref"][:,1] * aux_dict["line_delay"].squeeze()
    y_coord_obs_t_ns = aux_dict["meas_obs"][:,1] * aux_dict["line_delay"].squeeze()

    u_ld_ref_so3 = y_coord_ref_t_ns * aux_dict["inv_dt_so3"] + aux_dict["u_so3_ref"].squeeze()
    u_ld_ref_r3 = y_coord_ref_t_ns * aux_dict["inv_dt_r3"] + aux_dict["u_r3_ref"].squeeze()

    u_ld_obs_so3 = y_coord_obs_t_ns * aux_dict["inv_dt_so3"] + aux_dict["u_so3_obs"].squeeze()
    u_ld_obs_r3 = y_coord_obs_t_ns * aux_dict["inv_dt_r3"] + aux_dict["u_so3_obs"].squeeze()

    # evalute reference rolling shutter pose
    R_w_i_ref = spline_helper.evaluate_lie_vec(
        knots_se3[:4,:,:3,:3], u_ld_ref_so3, 
        aux_dict["inv_dt_so3"], derivatives=0, num_meas=num_fs)[0]

    t_w_i_ref = spline_helper.evaluate_euclidean_vec(
        knots_se3[:4,:,:3,3], u_ld_ref_r3, 
        aux_dict["inv_dt_r3"], derivatives=0, num_meas=num_fs)

    R_w_i_obs = spline_helper.evaluate_lie_vec(
        knots_se3[4:,:,:3,:3], u_ld_obs_so3, 
        aux_dict["inv_dt_so3"], derivatives=0, num_meas=num_fs)[0]

    t_w_i_obs = spline_helper.evaluate_euclidean_vec(
        knots_se3[4:,:,:3,3], u_ld_obs_r3, 
        aux_dict["inv_dt_r3"], derivatives=0, num_meas=num_fs)

    # project point to camera
    depths = 1. / aux_dict["inv_depth"]
    bearings_scaled = depths.unsqueeze(-1) * aux_dict["ref_bearings"]

    T_i_c = th.SE3(tensor=aux_dict["T_i_c"])
    # 1. convert point from bearing vector to 3d point using
    # inverse depth from reference view and transform from camera to IMU
    # reference frame
    X_ref = T_i_c.transform_to(bearings_scaled)
    # 2. Transform point from IMU to world frame
    X = R_w_i_ref.rotate(X_ref) + th.Point3(tensor=t_w_i_ref)
    # 3. Transform point from world to IMU reference frame at observation   Vector3 X = R_ref_w_i * X_ref + t_ref_w_i;
    X_obs = R_w_i_obs.inverse().rotate(X - th.Point3(t_w_i_obs))
    # 4. Transform point from IMU reference frame to camera frame
    X_camera = T_i_c.inverse().transform_to(X_obs)

    x_camera = aux_dict["cam_matrix"] @ X_camera.tensor.unsqueeze(-1)

    return x_camera[:,:2,0] / x_camera[:,2] - aux_dict["meas_obs"]