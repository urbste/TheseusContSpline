import torch
import torch.nn as nn
import numpy as np
import time

import theseus as th
from theseus.core.cost_function import ScaleCostWeight

from so3_spline import SO3Spline
from rd_spline import RDSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from spline_helper import SplineHelper
import time_util
import time

class SplineEstimator3D(nn.Module):
    def __init__(self, N, dt_ns_so3, dt_ns_r3, T_i_c, cam_matrix):
        super().__init__()
        self.N = N
        self.device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

        self.so3_spline = SO3Spline(0, 0, dt_ns=dt_ns_so3, N=N, device=self.device)
        self.r3_spline = RDSpline(0, 0, dt_ns=dt_ns_r3, dim=3, N=N, device=self.device)

        self.r3_knots_in_problem = []
        self.so3_knots_in_problem = []

        # init some variable
        self.dt_ns_so3 = th.Variable(
            tensor=torch.tensor(dt_ns_so3).float().unsqueeze(0).to(self.device), 
            name="dt_ns_so3")
        self.dt_ns_r3 = th.Variable(
            tensor=torch.tensor(dt_ns_r3).float().unsqueeze(0).to(self.device), 
            name="dt_ns_r3")

        self.inv_dt_so3 = th.Variable(
            tensor=torch.tensor(time_util.S_TO_NS/dt_ns_so3).float().unsqueeze(0).to(self.device), 
            name="inv_dt_so3")
        self.inv_dt_r3 = th.Variable(
            tensor=torch.tensor(time_util.S_TO_NS/dt_ns_r3).float().unsqueeze(0).to(self.device), 
            name="inv_dt_r3")

        self.gravity = th.Variable(
            torch.tensor([0., 0., -9.81]).float().to(self.device), 
            name="gravity")

        self.objective = th.Objective()
        self.theseus_inputs = {}

        self.T_i_c = th.SE3(
            tensor=T_i_c.float().unsqueeze(0).to(self.device), 
            name="T_i_c")
        self.T_c_i = self.T_i_c.inverse()

        self.line_delay = th.Variable(
            tensor=torch.tensor([0.0]).float().unsqueeze(0).to(self.device), 
            name="line_delay")
        self.cam_matrix = th.Variable(
            tensor=torch.tensor(cam_matrix).float().unsqueeze(0).to(self.device), 
            name="cam_matrix")

        self.spline_helper = SplineHelper(N, self.device)
        
        self.cnt_repro_err = 0

        self.robust_loss = th.HuberLoss()

    def reinit_spline_with_times(self, start_ns, end_ns):
        self.so3_spline.start_time_ns = start_ns
        self.r3_spline.start_time_ns = start_ns
        self.so3_spline.end_time_ns = end_ns
        self.r3_spline.end_time_ns = end_ns

        duration = end_ns-start_ns
        self.num_knots_so3 = (duration / self.dt_ns_so3.tensor[0] + self.N).int()
        self.num_knots_r3 = (duration / self.dt_ns_r3.tensor[0] + self.N).int()

    # input theia reconstruction
    def init_spline_with_vision(self, recon):

        # first gather times of trajectory
        vids = sorted(recon.ViewIds)
        cam_timestamps = []
        q_map, t_map = [], []
        for v in vids:
            t_s = recon.View(v).GetTimestamp()
            cam_timestamps.append(t_s)
            cam = recon.View(v).Camera()
            R_w_c = torch.tensor(cam.GetOrientationAsRotationMatrix().T).float().to(self.device)
            t_w_c = torch.tensor(cam.GetPosition()).float().to(self.device)

            T_w_c = th.SE3()
            T_w_c.update_from_rot_and_trans(
                rotation=th.SO3(tensor=R_w_c.unsqueeze(0)), 
                translation=th.Point3(tensor=t_w_c.unsqueeze(0))
            )
            # imu to world transformation
            T_w_i = T_w_c.compose(self.T_i_c.inverse())

            q_map.append(R.from_matrix(T_w_i.rotation().tensor.cpu().squeeze(0).numpy()).as_quat())
            t_map.append(T_w_i.translation().tensor.cpu().squeeze(0).numpy())

        self.reinit_spline_with_times(
            cam_timestamps[0]*time_util.S_TO_NS,
            cam_timestamps[-1]*time_util.S_TO_NS)

        # get spline times
        t_so3_spline, t_r3_spline = [], []
        for i in range(self.num_knots_so3):
            t_ = (i*self.dt_ns_so3.tensor.cpu().numpy()[0] + self.so3_spline.start_time_ns)*time_util.NS_TO_S
            if t_ >= cam_timestamps[-1]:
                t_so3_spline.append(t_so3_spline[-1])
            else:
                t_so3_spline.append(t_)
        for i in range(self.num_knots_r3):
            t_ = (i*self.dt_ns_r3.tensor.cpu().numpy()[0] + self.r3_spline.start_time_ns)*time_util.NS_TO_S
            if t_ >= cam_timestamps[-1]:
                t_r3_spline.append(t_r3_spline[-1])
            else:
                t_r3_spline.append(t_)
               
        # interpolate vision times at spline times
        cam_ts_np = np.array(cam_timestamps)
        t_map_np = np.array(t_map)
        slerp = Slerp(cam_ts_np, R.from_quat(q_map))
        interp_rots = slerp(t_so3_spline)
        tx = np.interp(t_r3_spline, cam_ts_np, t_map_np[:,0])
        ty = np.interp(t_r3_spline, cam_ts_np, t_map_np[:,1])
        tz = np.interp(t_r3_spline, cam_ts_np, t_map_np[:,2])

        # interpolate translation
        for i in range(len(interp_rots)):
            R_tensor = torch.tensor(interp_rots[i].as_matrix()).float().unsqueeze(0)
            self.so3_spline.knots.append(
                th.SO3(tensor=R_tensor.to(self.device), name="so3_knot_"+str(i)))

        for i in range(tx.shape[0]):
            t_tensor = torch.tensor(np.array([tx[i],ty[i],tz[i]])).float().unsqueeze(0)
            self.r3_spline.knots.append(
                th.Vector(tensor=t_tensor.to(self.device), 
                name="r3_knot_"+str(i)))

        # add optim variables --> all spline knots
        self.optim_vars = []
        for i in range(len(self.r3_spline.knots)):
            self.optim_vars.append(self.r3_spline.knots[i])
            self.r3_knots_in_problem.append(False)
            
        for i in range(len(self.so3_spline.knots)):
            self.optim_vars.append(self.so3_spline.knots[i])
            self.so3_knots_in_problem.append(False)

    def _calc_time_so3(self, sensor_time_ns):
        return time_util.calc_times(
            sensor_time_ns,
            self.so3_spline.start_time_ns,
            self.so3_spline.dt_ns,
            len(self.so3_spline.knots),
            self.N)

    def _calc_time_r3(self, sensor_time_ns):
        return time_util.calc_times(
            sensor_time_ns,
            self.r3_spline.start_time_ns,
            self.r3_spline.dt_ns,
            len(self.r3_spline.knots),
            self.N)

    def _rs_error(self, optim_vars, aux_vars):

        start = time.time()
        
        so3_knots = optim_vars[:len(optim_vars)//2]
        r3_knots = optim_vars[len(optim_vars)//2:]
        
        indices = aux_vars[0].tensor
        u = aux_vars[1].tensor
        bearings = aux_vars[2].tensor
        inv_depths = aux_vars[3].tensor
        ref_obs = aux_vars[4].tensor
        obs_obs = aux_vars[5].tensor
        # (obs, N, )
        s_so3_ref =  indices[:,:,0].int().flatten()
        s_so3_obs = indices[:,:,1].int().flatten()
        s_r3_ref = indices[:,:,2].int().flatten()
        s_r3_obs = indices[:,:,3].int().flatten()
        num_obs = indices.shape[0]

        u_so3_ref = u[:,0]
        u_r3_ref = u[:,1]
        u_so3_obs = u[:,2]
        u_r3_obs = u[:,3]

        # num_obs = len(inv_depths)
        # knots_se3 = opt_dict["knots"].reshape(sdim, num_fs, 3, 4)
        all_R_refs = torch.cat([so3_knots[idx].tensor[0] for idx in s_so3_ref],0).reshape(self.N,num_obs,3,3)
        all_p_refs = torch.cat([r3_knots[idx].tensor[0] for idx in s_r3_ref],0).reshape(self.N,num_obs,3) 
        all_R_obs = torch.cat([so3_knots[idx].tensor[0] for idx in s_so3_obs],0).reshape(self.N,num_obs,3,3) 
        all_p_obs = torch.cat([r3_knots[idx].tensor[0] for idx in s_r3_obs],0).reshape(self.N,num_obs,3) 

        y_coord_ref_t_ns = ref_obs[:,1] * self.line_delay.tensor[0]
        y_coord_obs_t_ns = obs_obs[:,1] * self.line_delay.tensor[0]

        u_ld_ref_so3 = y_coord_ref_t_ns * self.inv_dt_so3.tensor[0] + u_so3_ref
        u_ld_ref_r3 = y_coord_ref_t_ns * self.inv_dt_r3.tensor[0] + u_r3_ref

        u_ld_obs_so3 = y_coord_obs_t_ns * self.inv_dt_so3.tensor[0] + u_so3_obs
        u_ld_obs_r3 = y_coord_obs_t_ns * self.inv_dt_r3.tensor[0] + u_r3_obs
        # evalute reference rolling shutter pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            all_R_refs, u_ld_ref_so3, 
            self.inv_dt_so3.tensor, derivatives=0, num_meas=num_obs)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            all_p_refs, u_ld_ref_r3, 
            self.inv_dt_r3.tensor, derivatives=0, num_meas=num_obs)

        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            all_R_obs, u_ld_obs_so3, 
            self.inv_dt_so3.tensor, derivatives=0, num_meas=num_obs)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            all_p_obs, u_ld_obs_r3, 
            self.inv_dt_r3.tensor, derivatives=0, num_meas=num_obs)

        # project point to camera
        depths = 1. / inv_depths
        bearings_scaled = depths * bearings

        # 1. convert point from bearing vector to 3d point using
        # inverse depth from reference view and transform from camera to IMU
        # reference frame
        X_ref = self.T_i_c.transform_to(bearings_scaled.squeeze(0))
        # 2. Transform point from IMU to world frame
        X = R_w_i_ref.rotate(X_ref) + th.Point3(tensor=t_w_i_ref)
        # 3. Transform point from world to IMU reference frame at observation   Vector3 X = R_ref_w_i * X_ref + t_ref_w_i;
        X_obs = R_w_i_obs.inverse().rotate(X - th.Point3(t_w_i_obs))
        # 4. Transform point from IMU reference frame to camera frame
        X_camera = self.T_c_i.transform_to(X_obs)

        x_camera = self.cam_matrix.tensor @ X_camera.tensor.unsqueeze(-1)

        repro_error = x_camera[:,:2,0] / x_camera[:,2] - obs_obs
        print("Mean reprojection error: {:.3f}".format(torch.mean(repro_error)))
        print("Number of residuals: ",repro_error.shape[0])
        print("Time to eval residuals: {:.3f}s".format(time.time()-start))

        return x_camera[:,:2,0] / x_camera[:,2] - obs_obs

    def add_rs_view(self, view, view_id, recon):
        
        #with torch.no_grad():
        # iterate observations of that view
        tracks = view.TrackIds()

        aux_vars = []

        # pass knot ids to optimizer,
        # we unfold those in the cost function
        added_res_for_view = 0
        for t_idx_loop, t_id in enumerate(tracks):
            ref_view_id = recon.Track(t_id).ReferenceViewId()
            ref_view = recon.View(ref_view_id)

            img_obs_time_ns = time_util.S_TO_NS * (
                view.GetTimestamp() + self.line_delay.tensor[0]*view.GetFeature(t_id).point[1])
            img_ref_time_ns = time_util.S_TO_NS * (
                ref_view.GetTimestamp() + self.line_delay.tensor[0]*ref_view.GetFeature(t_id).point[1])
            
            # if ref and obs id are the same, inverse depth can not be estimated
            if img_obs_time_ns == img_ref_time_ns:
                continue

            u_so3_obs, s_so3_obs, suc1 = self._calc_time_so3(img_obs_time_ns)
            u_r3_obs, s_r3_obs, suc2 = self._calc_time_r3(img_obs_time_ns)

            u_so3_ref, s_so3_ref, suc3 = self._calc_time_so3(img_ref_time_ns)
            u_r3_ref, s_r3_ref, suc4 = self._calc_time_r3(img_ref_time_ns)

            suc = suc1 and suc2 and suc3 and suc4
            if not suc:
                print("time calc failed")
                continue
        
            knot_ids = torch.zeros((1,self.N,4)).int()
            knot_us = torch.zeros((1,4)).int()
            bearings = torch.zeros((1,3)).float()
            inv_depths = torch.zeros((1,1)).float()
            ref_obs = torch.zeros((1,2)).float()
            obs_obs = torch.zeros((1,2)).float()

            # create knots id lists. starting from s_* until s_*+self.N
            knot_ids_ref_so3 = list(range(s_so3_ref,s_so3_ref+self.N))
            knot_ids_obs_so3 = list(range(s_so3_obs,s_so3_obs+self.N))
            knot_ids_ref_r3 = list(range(s_r3_ref,s_r3_ref+self.N))
            knot_ids_obs_r3 = list(range(s_r3_obs,s_r3_obs+self.N))

            # keep track to add the global knot ids to the theseus_input dict
            for i in range(self.so3_spline.N):
                self.r3_knots_in_problem[s_so3_ref + i] = True
                self.so3_knots_in_problem[s_r3_ref + i] = True
                self.r3_knots_in_problem[s_so3_obs + i] = True
                self.so3_knots_in_problem[s_r3_obs + i] = True

            # get all knots that are in both reference and observing camera
            knots_in_cost_ids_so3 = sorted(set(knot_ids_ref_so3) | set(knot_ids_obs_so3))
            knots_in_cost_ids_r3 = sorted(set(knot_ids_ref_r3) | set(knot_ids_obs_r3))

            # get start knots for each spline (N=4) and camera
            # e.g. global cam_ref_knots_ids = [10,11,12,13], cam_obs_knots_ids=[12,13,14,15]
            # then global knots_in_cost are [10,11,12,13,14,15]
            # however in each cost locally the knot ids are [0,1,2,3,4,5]
            # hence in the cost function the start knots for cam_ref will be 0 -> [0,1,2,3]
            # and in the cost function the start knots for cam_obs will be 2 -> [2,3,4,5]
            # in turn this is split in so3 and r3 spline as they do not necessarily have the same dt 
            # and thus not the same global ids
            start_idx_ref_so3 = knots_in_cost_ids_so3.index(knot_ids_ref_so3[0])
            start_idx_obs_so3 = knots_in_cost_ids_so3.index(knot_ids_obs_so3[0])
            start_idx_ref_r3 = knots_in_cost_ids_r3.index(knot_ids_ref_r3[0])
            start_idx_obs_r3 = knots_in_cost_ids_r3.index(knot_ids_obs_r3[0])

            knot_start_ids = torch.arange(0,4).repeat(1,4).reshape(4,4).T + torch.tensor(
                [[[start_idx_ref_so3,start_idx_obs_so3,start_idx_ref_r3, start_idx_obs_r3]]])

            optim_vars = [self.so3_spline.knots[idx] for idx in knots_in_cost_ids_so3]
            optim_vars.extend([self.r3_spline.knots[idx] for idx in knots_in_cost_ids_r3])
            
            # get local indices of knots
            knot_us[0,0] = u_so3_ref
            knot_us[0,1] = u_r3_ref
            knot_us[0,2] = u_so3_obs
            knot_us[0,3] = u_r3_obs

            bearings[0,:] = torch.tensor(
                recon.Track(t_id).ReferenceBearingVector()).float().unsqueeze(0).to(self.device)
            inv_depths[0,:] = torch.tensor(
                recon.Track(t_id).InverseDepth()).float().unsqueeze(0).to(self.device)
            ref_obs[0,:] = torch.tensor(
                ref_view.GetFeature(t_id).point).float().unsqueeze(0).to(self.device)
            obs_obs[0,:] = torch.tensor(
                view.GetFeature(t_id).point).float().unsqueeze(0).to(self.device)

            aux_vars = [
                th.Variable(tensor=knot_start_ids.float(), name="knot_ids_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=knot_us.float(), name="knot_us_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=bearings.float(), name="bearings_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=inv_depths.float(), name="inv_depths_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=ref_obs.float(), name="ref_obs_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=obs_obs.float(), name="obs_obs_"+str(view_id)+"_"+str(t_idx_loop)),
            ]

            cost_function = th.AutoDiffCostFunction(
                optim_vars, 
                self._rs_error, 
                2, 
                aux_vars=aux_vars,
                name="rs_repro_cost_"+str(view_id)+"_"+str(t_idx_loop), 
                autograd_vectorize=False, 
                autograd_mode=th.AutogradMode.DENSE,
                autograd_strict=False)

            # robust_cost_function = th.RobustCostFunction(
            #     cost_function,
            #     th.HuberLoss,
            #     torch.tensor(3.0).float().to(self.device))
            
            self.objective.add(cost_function)

            self.cnt_repro_err += 1
            added_res_for_view += 1
        print("Added {} residuals for view {}".format(added_res_for_view,view_id))

    def init_optimizer(self):
        self.spline_optimizer = th.LevenbergMarquardt(
            self.objective,
            max_iterations=15,
            step_size=0.5,
            vectorize=False,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver
        )
        self.spline_optimizer_layer = th.TheseusLayer(self.spline_optimizer)
        self.spline_optimizer_layer.to(self.device)

    def forward(self):
        # get inputs
        for idx, knot in enumerate(self.r3_spline.knots):
            if self.r3_knots_in_problem[idx]:
                self.theseus_inputs[knot.name] = knot
        for idx, knot in enumerate(self.so3_spline.knots):
            if self.so3_knots_in_problem[idx]:
                self.theseus_inputs[knot.name] = knot

        sol, info = self.spline_optimizer_layer.forward(
            self.theseus_inputs, optimizer_kwargs={
                "damping": 1.0, 
                "track_best_solution": True, 
                "verbose": True}
        )
        print("Optim error: ", info.last_err.item())
        return sol

import pytheia as pt
recon = pt.io.ReadReconstruction("spline_recon_run1.recon")[1]

cam_matrix = recon.View(0).Camera().GetCalibrationMatrix()

T_i_c = torch.eye(3,4).float()
est = SplineEstimator3D(4, 0.1*time_util.S_TO_NS, 
    0.1*time_util.S_TO_NS, T_i_c, cam_matrix)
est.init_spline_with_vision(recon)

for v in sorted(recon.ViewIds)[1:]:
    est.add_rs_view(recon.View(v),  v, recon)
    print("added view: ",v)
    if v > 3:
        break

est.init_optimizer()
est.forward()


