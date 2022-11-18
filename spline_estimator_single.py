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
from reprojection_errors import RSError

class SplineEstimator3D(nn.Module):
    def __init__(self, N, dt_ns_so3, dt_ns_r3, T_i_c, cam_matrix, device):
        super().__init__()
        self.N = N
        self.device = device 

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
            tensor=torch.tensor([1/480*1/50]).float().unsqueeze(0).to(self.device), 
            name="line_delay")
        self.cam_matrix = th.Variable(
            tensor=torch.tensor(cam_matrix).float().unsqueeze(0).to(self.device), 
            name="cam_matrix")

        self.spline_helper = SplineHelper(N, self.device)
        
        self.cnt_repro_err = 0
        self.repro_cost_weight = ScaleCostWeight(scale=torch.tensor(1.0).float().to(self.device),
            name="repro_cost_weight")

        self.robust_loss_cls = th.HuberLoss

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


    def add_rs_view(self, view, view_id, recon, optim_inv_depth=True):
        
        #with torch.no_grad():
        # iterate observations of that view
        tracks = view.TrackIds()

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
            knot_us = torch.tensor([u_so3_ref, u_r3_ref, u_so3_obs, u_r3_obs]).unsqueeze(0)
            bearings = torch.tensor(
                recon.Track(t_id).ReferenceBearingVector()).float().unsqueeze(0).to(self.device)
            inv_depths = torch.tensor(
                recon.Track(t_id).InverseDepth()).float().unsqueeze(0).to(self.device)
            ref_obs = torch.tensor(
                ref_view.GetFeature(t_id).point).float().unsqueeze(0).to(self.device)
            obs_obs = torch.tensor(
                view.GetFeature(t_id).point).float().unsqueeze(0).to(self.device)

            aux_vars = [
                th.Variable(tensor=knot_us.float().to(self.device), name="knot_us_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=bearings.float().to(self.device), name="bearings_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=inv_depths.float().to(self.device), name="inv_depths_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=ref_obs.float().to(self.device), name="ref_obs_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=obs_obs.float().to(self.device), name="obs_obs_"+str(view_id)+"_"+str(t_idx_loop)),
            ]

            cost_function = th.AutoDiffCostFunction(
                optim_vars, 
                RSError(knot_start_ids.squeeze(0),
                        self.line_delay, [self.inv_dt_so3, self.inv_dt_r3], 
                        self.T_i_c, self.cam_matrix), 
                2, 
                aux_vars=aux_vars,
                cost_weight=self.repro_cost_weight,
                name="rs_repro_cost_"+str(view_id)+"_"+str(t_idx_loop), 
                autograd_vectorize=True, 
                autograd_mode=th.AutogradMode.VMAP,
                autograd_strict=False)

            log_loss_radius = th.Vector(
                tensor=torch.tensor(5).float().to(self.device).unsqueeze(0),
                name="log_loss_radius"+str(view_id)+"_"+str(t_idx_loop))

            robust_cost_function = th.RobustCostFunction(
                cost_function,
                self.robust_loss_cls,
                log_loss_radius=log_loss_radius)

            self.objective.add(robust_cost_function)

            self.cnt_repro_err += 1
            added_res_for_view += 1
        
        print("Added {} residuals for view {}".format(added_res_for_view,view_id))

    def init_optimizer(self):
        self.spline_optimizer = th.LevenbergMarquardt(
            self.objective,
            max_iterations=15,
            step_size=0.5,
            vectorize=True,
            linearization_cls=th.SparseLinearization,
            linear_solver_cls=th.CholmodSparseSolver if self.device == "cpu" else th.LUCudaSparseSolver
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
                "damping": 0.1, 
                "track_best_solution": True, 
                "verbose": True}
        )
        print("Optim error: ", info.last_err.item())
        return sol

import pytheia as pt
recon = pt.io.ReadReconstruction("spline_recon_run1.recon")[1]

cam_matrix = recon.View(0).Camera().GetCalibrationMatrix()


device = "cuda" if torch.cuda.is_available() else "cpu"
T_i_c = th.SE3(x_y_z_quaternion=torch.tensor([[
    0.013991958832708196,-0.040766470166917895,0.01589418686420154,
    0.0017841953862121206,-0.0014240956361964555,-0.7055056377782172,0.7087006304932949]])).tensor.squeeze()

est = SplineEstimator3D(4, 0.1*time_util.S_TO_NS, 
    0.1*time_util.S_TO_NS, T_i_c, cam_matrix, device)
est.init_spline_with_vision(recon)

for v in sorted(recon.ViewIds)[1:]:
    est.add_rs_view(recon.View(v),  v, recon)
    print("added view: ",v)
    if v > 10:
        break

est.init_optimizer()
est.forward()


