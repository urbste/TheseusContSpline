import torch
import theseus as th
from cost_functions import rs_reproj_error
from so3_spline import SO3Spline
from rd_spline import RDSpline
import time_util
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import time
import torch
import torch.nn as nn


class SplineEstimator3D(nn.Module):
    def __init__(self, N, dt_ns_so3, dt_ns_r3, T_i_c, cam_matrix):
        self.N = N
        self.so3_spline = SO3Spline(0, 0, dt_ns=dt_ns_so3, N=N)
        self.r3_spline = RDSpline(0, 0, dt_ns=dt_ns_r3, dim=3, N=N)

        # init some variable
        self.dt_ns_so3 = th.Variable(tensor=torch.tensor(dt_ns_so3).float().unsqueeze(0), name="dt_ns_so3")
        self.dt_ns_r3 = th.Variable(tensor=torch.tensor(dt_ns_r3).float().unsqueeze(0), name="dt_ns_r3")

        self.inv_dt_so3 = th.Variable(tensor=torch.tensor(time_util.S_TO_NS/dt_ns_so3).float().unsqueeze(0), name="inv_dt_so3")
        self.inv_dt_r3 = th.Variable(tensor=torch.tensor(time_util.S_TO_NS/dt_ns_r3).float().unsqueeze(0), name="inv_dt_r3")

        self.gravity = th.Variable(torch.tensor([0., 0., -9.81]).float(), name="gravity")

        self.objective = th.Objective()

        self.T_i_c = th.SE3(tensor=T_i_c.float().unsqueeze(0), name="T_i_c")
        self.line_delay = th.Variable(tensor=torch.tensor([0.0]).float().unsqueeze(0), name="line_delay")
        self.cam_matrix = th.Variable(tensor=torch.tensor(cam_matrix).float().unsqueeze(0), name="cam_matrix")

        self.cnt_repro_err = 0

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
            R_w_c = torch.tensor(cam.GetOrientationAsRotationMatrix().T).float()
            t_w_c = torch.tensor(cam.GetPosition()).float()

            T_w_c = th.SE3()
            T_w_c.update_from_rot_and_trans(
                rotation=th.SO3(tensor=R_w_c.unsqueeze(0)), 
                translation=th.Point3(tensor=t_w_c.unsqueeze(0))
            )
            # imu to world transformation
            T_w_i = T_w_c.compose(self.T_i_c.inverse())

            q_map.append(R.from_matrix(T_w_i.rotation().tensor.squeeze(0)).as_quat())
            t_map.append(T_w_i.translation().tensor.squeeze(0).numpy())

        self.reinit_spline_with_times(
            cam_timestamps[0]*time_util.S_TO_NS,
            cam_timestamps[-1]*time_util.S_TO_NS)

        # get spline times
        t_so3_spline, t_r3_spline = [], []
        for i in range(self.num_knots_so3):
            t_ = (i*self.dt_ns_so3.tensor.numpy()[0] + self.so3_spline.start_time_ns)*time_util.NS_TO_S
            if t_ >= cam_timestamps[-1]:
                t_so3_spline.append(t_so3_spline[-1])
            else:
                t_so3_spline.append(t_)
        for i in range(self.num_knots_r3):
            t_ = (i*self.dt_ns_r3.tensor.numpy()[0] + self.r3_spline.start_time_ns)*time_util.NS_TO_S
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
            self.so3_spline.knots.append(th.SO3(tensor=R_tensor, name="so3_knot_"+str(i)))

        for i in range(tx.shape[0]):
            t_tensor = torch.tensor(np.array([tx[i],ty[i],tz[i]])).float().unsqueeze(0)
            self.r3_spline.knots.append(th.Vector(tensor=t_tensor, name="r3_knot_"+str(i)))

        # add optim variables --> all spline knots
        self.optim_vars = []
        for i in range(len(self.r3_spline.knots)):
            self.optim_vars.append(self.r3_spline.knots[i])

            knot = self.r3_spline.knots[i]
            self.theseus_inputs[knot.name] = self.r3_spline.knots[i].tensor

        for i in range(len(self.so3_spline.knots)):
            self.optim_vars.append(self.so3_spline.knots[i])

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

        r3_knots = optim_vars[:len(optim_vars)//2]
        so3_knots = optim_vars[len(optim_vars)//2:]
        
        s = aux_vars[0].tensor
        u = aux_vars[1].tensor
        bearings = aux_vars[0].tensor
        inv_depths = aux_vars[1].tensor
        ref_obs = aux_vars[0].tensor
        ob_obs = aux_vars[1].tensor
        # (obs, N, )
        s_so3_ref = s[0,:,:,0].int()
        s_r3_ref = s[0,:,:,1].int()
        s_so3_obs = s[0,:,:,2].int()
        s_r3_obs = s[0,:,:,3].int()

        u_so3_ref = u[0,:,0]
        u_r3_ref = u[0,:,1]
        u_so3_obs = u[0,:,2]
        u_r3_obs = u[0,:,3]



        num_fs = len(u_r3_ref)
        knots_se3 = opt_dict["knots"].reshape(sdim, num_fs, 3, 4)


        y_coord_ref_t_ns = ref_obs[:,:,1] * self.line_delay.squeeze()
        y_coord_obs_t_ns = ob_obs[:,:,1] * self.line_delay.squeeze()

        u_ld_ref_so3 = y_coord_ref_t_ns * self.inv_dt_so3 + u_so3_ref.squeeze()
        u_ld_ref_r3 = y_coord_ref_t_ns * self.inv_dt_r3 + u_r3_ref.squeeze()

        u_ld_obs_so3 = y_coord_obs_t_ns * self.inv_dt_so3 + u_so3_obs.squeeze()
        u_ld_obs_r3 = y_coord_obs_t_ns * self.inv_dt_r3 + u_r3_obs.squeeze()

        # evalute reference rolling shutter pose
        R_w_i_ref = self.spline_helper.evaluate_lie_vec(
            knots_se3[:4,:,:3,:3], u_ld_ref_so3, 
            self.inv_dt_so3 , derivatives=0, num_meas=num_fs)[0]

        t_w_i_ref = self.spline_helper.evaluate_euclidean_vec(
            knots_se3[:4,:,:3,3], u_ld_ref_r3, 
            self.inv_dt_r3, derivatives=0, num_meas=num_fs)

        R_w_i_obs = self.spline_helper.evaluate_lie_vec(
            knots_se3[4:,:,:3,:3], u_ld_obs_so3, 
            self.inv_dt_so3 , derivatives=0, num_meas=num_fs)[0]

        t_w_i_obs = self.spline_helper.evaluate_euclidean_vec(
            knots_se3[4:,:,:3,3], u_ld_obs_r3, 
            self.inv_dt_r3, derivatives=0, num_meas=num_fs)

        # project point to camera
        depths = 1. / inv_depths
        bearings_scaled = depths.unsqueeze(-1) * bearings

        T_i_c = th.SE3(tensor=self.T_i_c)
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

        x_camera = self.cam_matrix @ X_camera.tensor.unsqueeze(-1)

        return x_camera[:,:2,0] / x_camera[:,2] - obs_obs

    def add_rs_view(self, view, recon):
        
        # iterate observations of that view
        tracks = view.TrackIds()

        opt_vars = []
        aux_vars = []
        str_cnt = view.Name()

        # pass knot ids to optimizer,
        # we unfold those in the cost function
        knot_ids = torch.zeros((1,len(tracks),self.N,4)).int()
        knot_us = torch.zeros((1,len(tracks),4)).int()
        bearings = torch.zeros((1,len(tracks),3)).float()
        inv_depths = torch.zeros((1,len(tracks),1)).float()
        ref_obs = torch.zeros((1,len(tracks),2)).float()
        obs_obs = torch.zeros((1,len(tracks),2)).float()

        for idx, t_id in enumerate(tracks):
            ref_view_id = recon.Track(t_id).ReferenceViewId()
            ref_view = recon.View(ref_view_id)

            img_obs_time_ns = torch.tensor(time_util.S_TO_NS * (
                view.GetTimestamp() + self.line_delay.tensor*view.GetFeature(t_id).point[1]))
            img_ref_time_ns = torch.tensor(time_util.S_TO_NS * (
                ref_view.GetTimestamp() + self.line_delay.tensor*ref_view.GetFeature(t_id).point[1]))

            u_so3_obs, s_so3_obs, suc1 = self._calc_time_so3(img_obs_time_ns)
            u_r3_obs, s_r3_obs, suc2 = self._calc_time_r3(img_obs_time_ns)

            u_so3_ref, s_so3_ref, suc3 = self._calc_time_so3(img_ref_time_ns)
            u_r3_ref, s_r3_ref, suc4 = self._calc_time_r3(img_ref_time_ns)

            suc = suc1 and suc2 and suc3 and suc4
            if not suc:
                print("time calc failed")
                continue

            optim_vars = []
            for i in range(self.so3_spline.N):
                knot_ids[0,idx,i,0] = s_so3_ref + i
                knot_ids[0,idx,i,1] = s_r3_ref + i
                knot_ids[0,idx,i,2] = s_so3_obs + i
                knot_ids[0,idx,i,3] = s_r3_obs + i
            knot_us[0,idx,0] = u_so3_ref
            knot_us[0,idx,1] = u_r3_ref
            knot_us[0,idx,2] = u_so3_obs
            knot_us[0,idx,3] = u_r3_obs

            bearings[0,idx,:] = torch.tensor(recon.Track(t_id).ReferenceBearingVector()).float().unsqueeze(0)
            inv_depths[0,idx,:] = torch.tensor(recon.Track(t_id).InverseDepth()).float().unsqueeze(0)
            ref_obs[0,idx,:] = torch.tensor(ref_view.GetFeature(t_id).point).float().unsqueeze(0)
            obs_obs[0,idx,:] = torch.tensor(view.GetFeature(t_id).point).float().unsqueeze(0)


        aux_vars = [
            th.Variable(tensor=knot_ids.float(), name="knot_ids_"+str_cnt),
            th.Variable(tensor=knot_us.float(), name="knot_us_"+str_cnt),
            th.Variable(tensor=bearings.float(), name="bearings_"+str_cnt),
            th.Variable(tensor=inv_depths.float(), name="bearings_"+str_cnt),
            th.Variable(tensor=ref_obs.float(), name="bearings_"+str_cnt),
            th.Variable(tensor=obs_obs.float(), name="bearings_"+str_cnt),
        ]

        cost_function = th.AutoDiffCostFunction(
            self.optim_vars, 
            self._rs_error, 
            2*len(tracks), 
            aux_vars=aux_vars, 
            name="rs_repro_cost_"+str_cnt, 
            autograd_vectorize=False, 
            autograd_mode=th.AutogradMode.LOOP_BATCH)

        self.objective.add(cost_function)

        self.cnt_repro_err += 1

    def init_optimizer(self):

        self.theseus_inputs = {}
        self.spline_optimizer = th.LevenbergMarquardt(
            self.objective,
            max_iterations=15,
            step_size=0.5,
        )
        self.spline_optimizer_layer = th.TheseusLayer(self.spline_optimizer)

    # Run theseus so that NN(x*) is close to y
    def forward(self, y):
        x0 = torch.ones(y.shape[0], 2)
        sol, info = self.layer.forward(
            {"x": x0, "y": y}, optimizer_kwargs={"damping": 0.1}
        )
        print("Optim error: ", info.last_err.item())
        return sol["x"]

import pytheia as pt
recon = pt.io.ReadReconstruction("/media/Data/Sparsenet/Ammerbach/Links/spline_recon_run1.recon")[1]

cam_matrix = recon.View(0).Camera().GetCalibrationMatrix()

est = SplineEstimator3D(4, 0.1*time_util.S_TO_NS, 
    0.1*time_util.S_TO_NS, torch.eye(3,4).float(), cam_matrix)
est.init_spline_with_vision(recon)

for v in sorted(recon.ViewIds)[2:]:
    est.add_rs_view(recon.View(v), recon)
    print("added ",v)
    if v > 5:
        break


est.optimize()