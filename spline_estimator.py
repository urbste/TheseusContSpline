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

class SplineEstimator3D:
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

        self.theseus_inputs = {}

    # def add_accelerometer(self, reading, time_ns):

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

        self.optim_vars = []
        for i in range(len(self.r3_spline.knots)):
            self.optim_vars.append(self.r3_spline.knots[i])

            knot = self.r3_spline.knots[i]
            self.theseus_inputs[knot.name] = self.r3_spline.knots[i].tensor

        for i in range(len(self.so3_spline.knots)):
            self.optim_vars.append(self.so3_spline.knots[i])

    # def add_gyroscope(self):
    #     u, s, suc = time_util.calc_times(time_pos_meas[k], self.so3_spline.start_time_ns, 
    #         self.so3_spline.dt_ns, len(self.so3_spline.knots), self.so3_spline.N)

    #     measurement = th.Variable(rot_m.tensor, name="gyro_meas_"+str(k))

    #     u = th.Variable(tensor=u.unsqueeze(0), name="u_gyro"+str(k))

    #     aux_vars = [measurement, u, inv_dt_s_]
    #     optim_vars = []

    #     for i in range(self.so3_spline.N):
    #         optim_vars.append(self.so3_spline.knots[i + s])

    #     cost_function = th.AutoDiffCostFunction(
    #         optim_vars, so3_error, DIM, aux_vars=aux_vars, name="pos_cost_"+str(k), 
    #         autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH
    #     )
    #     objective.add(cost_function)

    def calc_time_so3(self, sensor_time_ns):
        return time_util.calc_times(
            sensor_time_ns,
            self.so3_spline.start_time_ns,
            self.so3_spline.dt_ns,
            len(self.so3_spline.knots),
            self.N)

    def calc_time_r3(self, sensor_time_ns):
        return time_util.calc_times(
            sensor_time_ns,
            self.r3_spline.start_time_ns,
            self.r3_spline.dt_ns,
            len(self.r3_spline.knots),
            self.N)

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


        for idx, t_id in enumerate(tracks):
            ref_view_id = recon.Track(t_id).ReferenceViewId()
            ref_view = recon.View(ref_view_id)

            img_obs_time_ns = torch.tensor(time_util.S_TO_NS * (
                view.GetTimestamp() + self.line_delay.tensor*view.GetFeature(t_id).point[1]))
            img_ref_time_ns = torch.tensor(time_util.S_TO_NS * (
                ref_view.GetTimestamp() + self.line_delay.tensor*ref_view.GetFeature(t_id).point[1]))

            u_so3_obs, s_so3_obs, suc1 = self.calc_time_so3(img_obs_time_ns)
            u_r3_obs, s_r3_obs, suc2 = self.calc_time_r3(img_obs_time_ns)

            u_so3_ref, s_so3_ref, suc3 = self.calc_time_so3(img_ref_time_ns)
            u_r3_ref, s_r3_ref, suc4 = self.calc_time_r3(img_ref_time_ns)

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
  
            # create aux variable list
            # aux_vars.append
            #     th.Variable(
            #     th.Variable(torch.tensor(ref_view.GetFeature(t_id).point).float().unsqueeze(0), name="ref_obs_"+str_cnt), 
            #     th.Variable(torch.tensor(view.GetFeature(t_id).point).float().unsqueeze(0), name="obs_"+str_cnt), 
            #     th.Variable(tensor=u_r3_ref.unsqueeze(0), name="u_r3_ref_"+str_cnt),
            #     th.Variable(tensor=u_so3_ref.unsqueeze(0), name="u_so3_ref_"+str_cnt),
            #     th.Variable(tensor=u_r3_obs.unsqueeze(0), name="u_r3_obs_"+str_cnt),
            #     th.Variable(tensor=u_so3_obs.unsqueeze(0), name="u_so3_obs_"+str_cnt),
            #     self.inv_dt_r3,
            #     self.inv_dt_so3,
            #     th.Variable(torch.tensor(recon.Track(t_id).ReferenceBearingVector()).float().unsqueeze(0), name="ref_bear_"+str_cnt),
            #     self.T_i_c,
            #     self.line_delay,
            #     self.cam_matrix,
            #     th.Variable(torch.tensor(recon.Track(t_id).InverseDepth()).float().unsqueeze(0), name="inv_depth_"+str_cnt),
            # ]

        aux_vars = [
            th.Variable(tensor=knot_ids.float(), name="knot_ids_"+str_cnt),
            th.Variable(tensor=knot_us.float(), name="knot_us_"+str_cnt)
        ]

        cost_function = th.AutoDiffCostFunction(
            self.optim_vars, 
            rs_reproj_error, 
            2*len(tracks), 
            aux_vars=aux_vars, 
            name="rs_repro_cost_"+str_cnt, 
            autograd_vectorize=False, 
            autograd_mode=th.AutogradMode.LOOP_BATCH)

        self.objective.add(cost_function)

        self.cnt_repro_err += 1

    def optimize(self):
        optimizer = th.LevenbergMarquardt(
            self.objective,
            max_iterations=15,
            step_size=0.5,
        )
        theseus_optim = th.TheseusLayer(optimizer)

        with torch.no_grad():
            updated_inputs, info = theseus_optim.forward(
                self.theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})


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