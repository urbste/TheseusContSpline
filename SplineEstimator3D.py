# created by Steffen Urban, 11/2022
import torch
import torch.nn as nn
import numpy as np

import theseus as th
from theseus.core.cost_function import ScaleCostWeight

from spline.so3_spline import SO3Spline
from spline.rd_spline import RDSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from spline.spline_helper import SplineHelper
import spline.time_util as time_util
from residuals.camera import RollingShutterInvDepthRes, GlobalShutterInvDepthRes
from residuals.imu import GyroscopeRes, AccelerometerRes

torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)

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
            tensor=torch.tensor(dt_ns_so3).unsqueeze(0).to(self.device), 
            name="dt_ns_so3")
        self.dt_ns_r3 = th.Variable(
            tensor=torch.tensor(dt_ns_r3).unsqueeze(0).to(self.device), 
            name="dt_ns_r3")

        self.inv_dt_so3 = th.Variable(
            tensor=torch.tensor(time_util.S_TO_NS/dt_ns_so3).unsqueeze(0).to(self.device), 
            name="inv_dt_so3")
        self.inv_dt_r3 = th.Variable(
            tensor=torch.tensor(time_util.S_TO_NS/dt_ns_r3).unsqueeze(0).to(self.device), 
            name="inv_dt_r3")

        self.gravity = th.Variable(
            torch.tensor([0., 0., -9.81]).to(self.device), 
            name="gravity")

        self.objective = th.Objective()
        self.theseus_inputs = {}

        self.T_i_c = th.SE3(
            tensor=T_i_c.unsqueeze(0).to(self.device), 
            name="T_i_c")
        self.T_c_i = self.T_i_c.inverse()

        self.line_delay = th.Variable(
            tensor=torch.tensor([1/480*1/50]).unsqueeze(0).to(self.device), 
            name="line_delay")
        self.cam_matrix = th.Variable(
            tensor=torch.tensor(cam_matrix).unsqueeze(0).to(self.device), 
            name="cam_matrix")
        self.gravity = torch.tensor([0., 0., -9.81]).to(self.device)

        self.spline_helper = SplineHelper(N, self.device)
        
        # imu residuals
        self.cnt_gyro_res = 0
        self.cnt_accl_res = 0
        self.DIM_IMU_ERROR = 3

        # visual residuals
        self.DIM_VIS_ERROR = 2
        self.cnt_repro_err = 0
        self.repro_cost_weight = ScaleCostWeight(scale=torch.tensor(1.0).to(self.device),
            name="repro_cost_weight")
        self.robust_loss_cls = th.HuberLoss

    def set_gravity(self, gravity):
        self.gravity = gravity
    
    def reinit_spline_with_times(self, start_ns, end_ns):
        self.so3_spline.start_time_ns = start_ns
        self.r3_spline.start_time_ns = start_ns
        self.so3_spline.end_time_ns = end_ns
        self.r3_spline.end_time_ns = end_ns

        duration = end_ns-start_ns
        self.num_knots_so3 = (duration / self.dt_ns_so3.tensor[0] + self.N).int()
        self.num_knots_r3 = (duration / self.dt_ns_r3.tensor[0] + self.N).int()

    # input theia reconstruction
    def init_spline_with_vision(self, recon, max_time_s):

        # first gather times of trajectory
        vids = sorted(recon.ViewIds)
        cam_timestamps = []
        q_map, t_map = [], []
        for v in vids:
            t_s = recon.View(v).GetTimestamp()
            cam_timestamps.append(t_s)
            cam = recon.View(v).Camera()
            R_w_c = torch.tensor(cam.GetOrientationAsRotationMatrix().T).to(self.device)
            t_w_c = torch.tensor(cam.GetPosition()).to(self.device)

            T_w_c = th.SE3()
            T_w_c.update_from_rot_and_trans(
                rotation=th.SO3(tensor=R_w_c.unsqueeze(0)), 
                translation=th.Point3(tensor=t_w_c.unsqueeze(0))
            )
            # imu to world transformation
            T_w_i = T_w_c.compose(self.T_i_c.inverse())

            q_map.append(R.from_matrix(T_w_i.rotation().tensor.cpu().squeeze(0).numpy()).as_quat())
            t_map.append(T_w_i.translation().tensor.cpu().squeeze(0).numpy())
            if t_s > max_time_s:
                break
                
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
            R_tensor = torch.tensor(interp_rots[i].as_matrix()).unsqueeze(0)
            self.so3_spline.knots.append(
                th.SO3(tensor=R_tensor.to(self.device), name="so3_knot_"+str(i)))

        for i in range(tx.shape[0]):
            t_tensor = torch.tensor(np.array([tx[i],ty[i],tz[i]])).unsqueeze(0)
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

    def add_view(self, view, view_id, recon, robust_kernel_width=5., rolling_shutter=True):
        
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
                self.r3_knots_in_problem[s_r3_ref + i] = True
                self.so3_knots_in_problem[s_so3_ref + i] = True
                self.r3_knots_in_problem[s_r3_obs + i] = True
                self.so3_knots_in_problem[s_so3_obs + i] = True

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

            knot_start_ids = torch.arange(0,self.N).repeat(1,4).reshape(4,self.N).T + torch.tensor(
                [[[start_idx_ref_so3,start_idx_obs_so3,start_idx_ref_r3, start_idx_obs_r3]]])

            optim_vars = [self.so3_spline.knots[idx] for idx in knots_in_cost_ids_so3]
            start_id_r3 = len(optim_vars)
            optim_vars.extend([self.r3_spline.knots[idx] for idx in knots_in_cost_ids_r3])

            # get local indices of knots
            knot_us = torch.tensor([u_so3_ref, u_r3_ref, u_so3_obs, u_r3_obs]).unsqueeze(0)
            bearings = torch.tensor(
                recon.Track(t_id).ReferenceBearingVector()).unsqueeze(0).to(self.device)
            inv_depths = torch.tensor(
                recon.Track(t_id).InverseDepth()).unsqueeze(0).to(self.device)
            ref_obs = torch.tensor(
                ref_view.GetFeature(t_id).point).unsqueeze(0).to(self.device)
            obs_obs = torch.tensor(
                view.GetFeature(t_id).point).unsqueeze(0).to(self.device)

            aux_vars = [
                th.Variable(tensor=knot_us.to(self.device), name="knot_us_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=bearings.to(self.device), name="bearings_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=inv_depths.to(self.device), name="inv_depths_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=ref_obs.to(self.device), name="ref_obs_"+str(view_id)+"_"+str(t_idx_loop)),
                th.Variable(tensor=obs_obs.to(self.device), name="obs_obs_"+str(view_id)+"_"+str(t_idx_loop)),
            ]
            if rolling_shutter:
                cost_function = th.AutoDiffCostFunction(
                    optim_vars, 
                    RollingShutterInvDepthRes(knot_start_ids.squeeze(0),
                            self.line_delay, [self.inv_dt_so3, self.inv_dt_r3], 
                            self.T_i_c, self.cam_matrix, self.spline_helper, 
                            start_id_r3), 
                    self.DIM_VIS_ERROR, 
                    aux_vars=aux_vars,
                    cost_weight=self.repro_cost_weight,
                    name="rs_repro_cost_"+str(view_id)+"_"+str(t_idx_loop), 
                    autograd_vectorize=True, 
                    autograd_mode=th.AutogradMode.VMAP)
            else:
                cost_function = th.AutoDiffCostFunction(
                    optim_vars, 
                    GlobalShutterInvDepthRes(knot_start_ids.squeeze(0),
                            [self.inv_dt_so3, self.inv_dt_r3], 
                            self.T_i_c, self.cam_matrix, self.spline_helper, 
                            start_id_r3), 
                    self.DIM_VIS_ERROR, 
                    aux_vars=aux_vars,
                    cost_weight=self.repro_cost_weight,
                    name="rs_repro_cost_"+str(view_id)+"_"+str(t_idx_loop), 
                    autograd_vectorize=True, 
                    autograd_mode=th.AutogradMode.VMAP)

            log_loss_radius = th.Vector(
                tensor=torch.tensor(robust_kernel_width).to(self.device).unsqueeze(0),
                name="log_loss_radius"+str(view_id)+"_"+str(t_idx_loop))

            robust_cost_function = th.RobustCostFunction(
                cost_function,
                self.robust_loss_cls,
                log_loss_radius=log_loss_radius)

            self.objective.add(robust_cost_function)

            self.cnt_repro_err += 1
            added_res_for_view += 1
        
        print("Added {} residuals for view {}".format(added_res_for_view,view_id))
    
    def add_gyro_reading(self, reading, t_ns, weight):
        if t_ns < self.so3_spline.start_time_ns or \
        t_ns > self.so3_spline.end_time_ns:
            return
        u_so3_obs, s_so3_obs, suc1 = self._calc_time_so3(torch.tensor(t_ns))

        if not suc1:
            print("Error adding gyro residual for time {}".format(t_ns))
            return

        knot_us = u_so3_obs.unsqueeze(0)
        reading = torch.tensor(reading).unsqueeze(0)

        aux_vars = [th.Variable(tensor=knot_us.to(self.device), name="gyro_knot_us_"+str(self.cnt_gyro_res)),
                    th.Variable(tensor=reading.to(self.device), name="gyro_reading_"+str(self.cnt_gyro_res))]
        optim_vars = [self.so3_spline.knots[idx] for idx in range(s_so3_obs, s_so3_obs+self.N)]
        cost_function = th.AutoDiffCostFunction(
            optim_vars, 
            GyroscopeRes(self.inv_dt_so3, self.spline_helper), 
            self.DIM_IMU_ERROR, 
            aux_vars=aux_vars,
            cost_weight=self.repro_cost_weight,
            name="gyro_cost_"+str(self.cnt_gyro_res), 
            autograd_vectorize=True, 
            autograd_mode=th.AutogradMode.VMAP)

        self.objective.add(cost_function)
        self.cnt_gyro_res += 1

    def add_accel_reading(self, reading, t_ns, weight):
        if t_ns < self.so3_spline.start_time_ns or \
            t_ns > self.so3_spline.end_time_ns or \
            t_ns < self.r3_spline.start_time_ns or \
            t_ns > self.r3_spline.end_time_ns:
            return
        u_so3, s_so3, suc1 = self._calc_time_so3(torch.tensor(t_ns))
        u_r3, s_r3, suc2 = self._calc_time_r3(torch.tensor(t_ns))

        if not suc1 or not suc2:
            print("Error adding accel residual for time {}".format(t_ns))
            return

        knot_us = knot_us = torch.tensor([u_so3, u_r3]).unsqueeze(0)
        reading = torch.tensor(reading).unsqueeze(0)

        aux_vars = [th.Variable(tensor=knot_us.to(self.device), name="accl_knot_us_"+str(self.cnt_accl_res)),
                    th.Variable(tensor=reading.to(self.device), name="accl_reading_"+str(self.cnt_accl_res))]
        optim_vars = [self.so3_spline.knots[idx] for idx in range(s_so3, s_so3+self.N)]
        optim_vars.extend([self.r3_spline.knots[idx] for idx in range(s_r3, s_r3+self.N)])

        cost_function = th.AutoDiffCostFunction(
            optim_vars, 
            AccelerometerRes(self.inv_dt_so3, self.inv_dt_r3, 
                self.spline_helper, self.gravity), 
            self.DIM_IMU_ERROR, 
            aux_vars=aux_vars,
            cost_weight=self.repro_cost_weight,
            name="accl_cost_"+str(self.cnt_accl_res), 
            autograd_vectorize=True, 
            autograd_mode=th.AutogradMode.VMAP)

        self.objective.add(cost_function)
        self.cnt_accl_res += 1

    def add_gyroscope(self, gyroscope, sensor_times_ns, weighting):
        print("Adding gyroscope residuals.")
        for idx, t_ns in enumerate(sensor_times_ns):
            self.add_gyro_reading(gyroscope[idx], t_ns, weighting)
        print("Added {} gyro residuals.".format(self.cnt_gyro_res))
    
    def add_accelerometer(self, accelerometer, sensor_times_ns, weighting):
        print("Adding accelerometer residuals.")
        for idx, t_ns in enumerate(sensor_times_ns):
            self.add_accel_reading(accelerometer[idx], t_ns, weighting)
        print("Added {} accelerometer residuals.".format(self.cnt_accl_res))

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

