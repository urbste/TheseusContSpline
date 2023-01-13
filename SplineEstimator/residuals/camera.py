# created by Steffen Urban, 11/2022

import torch
import theseus as th

class InvDepthCamProjection:
    def __init__(self, T_i_c, cam_matrix):
        self.T_i_c = T_i_c
        self.T_c_i = T_i_c.inverse()
        self.cam_matrix = cam_matrix

    def project(self, bearings, inv_depths, 
            R_w_i_ref, t_w_i_ref,   
            R_w_i_obs, t_w_i_obs):
        # project point to camera
        bearings_scaled = bearings / inv_depths.unsqueeze(-1)

        # 1. convert point from bearing vector to 3d point using
        # inverse depth from reference view and transform from camera to IMU
        # reference frame
        X_ref = self.T_i_c.transform_from(bearings_scaled)
        # 2. Transform point from IMU to world frame
        X = R_w_i_ref.rotate(X_ref) + th.Point3(tensor=t_w_i_ref)
        # 3. Transform point from world to IMU reference frame at observation   Vector3 X = R_ref_w_i * X_ref + t_ref_w_i;
        X_obs = R_w_i_obs.inverse().rotate(X - th.Point3(tensor=t_w_i_obs))
        # 4. Transform point from IMU reference frame to camera frame
        X_camera = self.T_c_i.transform_from(X_obs)

        x_camera = self.cam_matrix.tensor @ X_camera.tensor.unsqueeze(-1)

        return x_camera[:,:2,0] / x_camera[:,2]

class GlobalShutterPoseRes:
    def __init__(self, knot_start_ids, inv_dts, T_i_c, cam_matrix, spline_helper, start_id_r3):

        self.so3_knot_idx_ref, self.so3_knot_idx_obs = knot_start_ids[:,0], knot_start_ids[:,1]
        self.r3_knot_idx_ref, self.r3_knot_idx_obs = knot_start_ids[:,2], knot_start_ids[:,3]

        self.inv_dt_so3, self.inv_dt_r3 = inv_dts[0], inv_dts[1]

        self.spline_helper = spline_helper
        # because number of so3 and r3 might be different
        self.start_id_r3 = start_id_r3

        self.cam = InvDepthCamProjection(T_i_c, cam_matrix)

    def __call__(self, optim_vars, aux_vars):
        so3_knots_ = optim_vars[:self.start_id_r3]  # each (B, 3, 3)
        r3_knots_ = optim_vars[self.start_id_r3:]   # each (B, 3)

        norm_times = aux_vars[0].tensor
        bearings = aux_vars[1].tensor
        obs_obs = aux_vars[2].tensor
        inv_depths = aux_vars[3].tensor

        u_so3_ref = norm_times[:,0]
        u_r3_ref = norm_times[:,1]
        u_so3_obs = norm_times[:,2]
        u_r3_obs = norm_times[:,3]

        # Shapes will be (4, B, 3, 3) and (4, B, 3)
        so3_knots_tensor_ref_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_ref], dim=0)
        so3_knots_tensor_obs_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_obs], dim=0)

        r3_knots_tensor_ref_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_ref], dim=0)
        r3_knots_tensor_obs_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_obs], dim=0)

        # evalute reference pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_ref_, u_so3_ref, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_ref_, u_r3_ref, 
            self.inv_dt_r3.tensor, derivatives=0)
            
        # evalute observing pose
        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_obs_, u_so3_obs, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_obs_, u_r3_obs, 
            self.inv_dt_r3.tensor, derivatives=0)

        # project point to camera
        x_camera = self.cam.project(bearings, inv_depths, 
            R_w_i_ref, t_w_i_ref, R_w_i_obs, t_w_i_obs)

        repro_error = x_camera - obs_obs

        return repro_error

class GlobalShutterInvDepthRes:
    def __init__(self, knot_start_ids, inv_dts, T_i_c, cam_matrix, spline_helper, start_id_r3, end_id_r3):

        self.so3_knot_idx_ref, self.so3_knot_idx_obs = knot_start_ids[:,0], knot_start_ids[:,1]
        self.r3_knot_idx_ref, self.r3_knot_idx_obs = knot_start_ids[:,2], knot_start_ids[:,3]

        self.inv_dt_so3, self.inv_dt_r3 = inv_dts[0], inv_dts[1]

        self.spline_helper = spline_helper
        # because number of so3 and r3 might be different
        self.start_id_r3 = start_id_r3
        self.end_id_r3 = end_id_r3

        self.cam = InvDepthCamProjection(T_i_c, cam_matrix)

    def __call__(self, optim_vars, aux_vars):
        so3_knots_ = optim_vars[:self.start_id_r3]  # each (B, 3, 3)
        r3_knots_ = optim_vars[self.start_id_r3:self.end_id_r3]   # each (B, 3)
        inv_depths = optim_vars[self.end_id_r3] # each (B, 1)

        norm_times = aux_vars[0].tensor
        bearings = aux_vars[1].tensor
        obs_obs = aux_vars[2].tensor

        u_so3_ref = norm_times[:,0]
        u_r3_ref = norm_times[:,1]
        u_so3_obs = norm_times[:,2]
        u_r3_obs = norm_times[:,3]

        # Shapes will be (4, B, 3, 3) and (4, B, 3)
        so3_knots_tensor_ref_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_ref], dim=0)
        so3_knots_tensor_obs_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_obs], dim=0)

        r3_knots_tensor_ref_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_ref], dim=0)
        r3_knots_tensor_obs_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_obs], dim=0)

        # evalute reference pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_ref_, u_so3_ref, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_ref_, u_r3_ref, 
            self.inv_dt_r3.tensor, derivatives=0)
            
        # evalute observing pose
        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_obs_, u_so3_obs, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_obs_, u_r3_obs, 
            self.inv_dt_r3.tensor, derivatives=0)

        # project point to camera
        x_camera = self.cam.project(bearings, inv_depths.tensor.squeeze(-1), 
            R_w_i_ref, t_w_i_ref, R_w_i_obs, t_w_i_obs)

        repro_error = x_camera - obs_obs

        return repro_error

class RollingShutterPoseRes:
    def __init__(self, knot_start_ids, line_delay, inv_dts, T_i_c, cam_matrix, spline_helper, start_id_r3):

        self.so3_knot_idx_ref, self.so3_knot_idx_obs = knot_start_ids[:,0], knot_start_ids[:,1]
        self.r3_knot_idx_ref, self.r3_knot_idx_obs = knot_start_ids[:,2], knot_start_ids[:,3]

        self.line_delay = line_delay
        self.inv_dt_so3, self.inv_dt_r3 = inv_dts[0], inv_dts[1]

        self.spline_helper = spline_helper
        # because number of so3 and r3 might be different
        self.start_id_r3 = start_id_r3
        
        self.cam = InvDepthCamProjection(T_i_c, cam_matrix)

    def __call__(self, optim_vars, aux_vars):
        so3_knots_ = optim_vars[:self.start_id_r3]  # each (B, 3, 3)
        r3_knots_ = optim_vars[self.start_id_r3:]   # each (B, 3)
        
        norm_times = aux_vars[0].tensor
        bearings = aux_vars[1].tensor
        obs_obs = aux_vars[2].tensor
        ref_obs = aux_vars[3].tensor
        inv_depths = aux_vars[4].tensor

        u_so3_ref = norm_times[:,0]
        u_r3_ref = norm_times[:,1]
        u_so3_obs = norm_times[:,2]
        u_r3_obs = norm_times[:,3]

        # Shapes will be (4, B, 3, 3) and (4, B, 3)
        so3_knots_tensor_ref_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_ref], dim=0)
        so3_knots_tensor_obs_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_obs], dim=0)

        r3_knots_tensor_ref_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_ref], dim=0)
        r3_knots_tensor_obs_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_obs], dim=0)

        y_coord_ref_t_s = ref_obs[:,1] * self.line_delay.tensor[0]
        y_coord_obs_t_s = obs_obs[:,1] * self.line_delay.tensor[0]

        u_ld_ref_so3 = y_coord_ref_t_s * self.inv_dt_so3.tensor[0] + u_so3_ref
        u_ld_ref_r3 = y_coord_ref_t_s * self.inv_dt_r3.tensor[0] + u_r3_ref

        u_ld_obs_so3 = y_coord_obs_t_s * self.inv_dt_so3.tensor[0] + u_so3_obs
        u_ld_obs_r3 = y_coord_obs_t_s * self.inv_dt_r3.tensor[0] + u_r3_obs
        
        # evaluate reference rolling shutter pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_ref_, u_ld_ref_so3, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_ref_, u_ld_ref_r3, 
            self.inv_dt_r3.tensor, derivatives=0)

        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_obs_, u_ld_obs_so3, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_obs_, u_ld_obs_r3, 
            self.inv_dt_r3.tensor, derivatives=0)

        x_camera = self.cam.project(bearings, inv_depths, 
            R_w_i_ref, t_w_i_ref, R_w_i_obs, t_w_i_obs)

        repro_error = x_camera - obs_obs

        return repro_error


class RollingShutterInvDepthRes:
    def __init__(self, knot_start_ids, line_delay, 
            inv_dts, T_i_c, cam_matrix, spline_helper, 
            start_id_r3, end_id_r3):

        self.so3_knot_idx_ref, self.so3_knot_idx_obs = knot_start_ids[:,0], knot_start_ids[:,1]
        self.r3_knot_idx_ref, self.r3_knot_idx_obs = knot_start_ids[:,2], knot_start_ids[:,3]

        self.line_delay = line_delay
        self.inv_dt_so3, self.inv_dt_r3 = inv_dts[0], inv_dts[1]

        self.spline_helper = spline_helper
        # because number of so3 and r3 might be different
        self.start_id_r3 = start_id_r3
        self.end_id_r3 = end_id_r3
        
        self.cam = InvDepthCamProjection(T_i_c, cam_matrix)

    def __call__(self, optim_vars, aux_vars):
        so3_knots_ = optim_vars[:self.start_id_r3]  # each (B, 3, 3)
        r3_knots_ = optim_vars[self.start_id_r3:self.end_id_r3]   # each (B, 3)
        inv_depths = optim_vars[self.end_id_r3]   # each (B, 1)

        norm_times = aux_vars[0].tensor
        bearings = aux_vars[1].tensor
        obs_obs = aux_vars[2].tensor
        ref_obs = aux_vars[3].tensor

        u_so3_ref = norm_times[:,0]
        u_r3_ref = norm_times[:,1]
        u_so3_obs = norm_times[:,2]
        u_r3_obs = norm_times[:,3]


        # Shapes will be (4, B, 3, 3) and (4, B, 3)
        so3_knots_tensor_ref_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_ref], dim=0)
        so3_knots_tensor_obs_ = torch.stack(
            [so3_knots_[i].tensor for i in self.so3_knot_idx_obs], dim=0)

        r3_knots_tensor_ref_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_ref], dim=0)
        r3_knots_tensor_obs_ = torch.stack(
            [r3_knots_[i].tensor for i in self.r3_knot_idx_obs], dim=0)

        y_coord_ref_t_ns = ref_obs[:,1] * self.line_delay.tensor[0]
        y_coord_obs_t_ns = obs_obs[:,1] * self.line_delay.tensor[0]

        u_ld_ref_so3 = y_coord_ref_t_ns * self.inv_dt_so3.tensor[0] + u_so3_ref
        u_ld_ref_r3 = y_coord_ref_t_ns * self.inv_dt_r3.tensor[0] + u_r3_ref

        u_ld_obs_so3 = y_coord_obs_t_ns * self.inv_dt_so3.tensor[0] + u_so3_obs
        u_ld_obs_r3 = y_coord_obs_t_ns * self.inv_dt_r3.tensor[0] + u_r3_obs

        # evalute reference rolling shutter pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_ref_, u_ld_ref_so3, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_ref_, u_ld_ref_r3, 
            self.inv_dt_r3.tensor, derivatives=0)

        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            so3_knots_tensor_obs_, u_ld_obs_so3, 
            self.inv_dt_so3.tensor, derivatives=0)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            r3_knots_tensor_obs_, u_ld_obs_r3, 
            self.inv_dt_r3.tensor, derivatives=0)

        x_camera = self.cam.project(bearings, inv_depths.tensor.squeeze(-1), 
            R_w_i_ref, t_w_i_ref, R_w_i_obs, t_w_i_obs)

        repro_error = x_camera - obs_obs

        return repro_error
