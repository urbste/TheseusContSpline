# created by Steffen Urban, 11/2022
import torch
import spline.time_util as time_util
import theseus as th
import pytheia as pt
import numpy as np
from util.telemetry_converter import TelemetryImporter
from SplineEstimator3D import SplineEstimator3D

# load telemetry
telemetry_importer = TelemetryImporter()
telemetry_importer.read_gopro_telemetry("test_data/run1.json")
recon = pt.io.ReadReconstruction("test_data/spline_recon_run1.recon")[1]

cam_matrix = recon.View(0).Camera().GetCalibrationMatrix()

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
T_i_c = th.SE3(x_y_z_quaternion=torch.tensor([[
    0.013991958832708196,-0.040766470166917895,0.01589418686420154,
    0.0017841953862121206,-0.0014240956361964555,-0.7055056377782172,0.7087006304932949]])).tensor.squeeze()

sorted_vids = sorted(recon.ViewIds)
max_id = 10
max_ts = recon.View(max_id).GetTimestamp()
spline_degree = 4
r3_dt_ns = 0.1 * time_util.S_TO_NS
so3_dt_ns = 0.07 * time_util.S_TO_NS

est = SplineEstimator3D(spline_degree, 
    so3_dt_ns, 
    r3_dt_ns, 
    T_i_c, cam_matrix, device)
est.init_spline_with_vision(recon, max_time_s=max_ts)

for v in sorted(recon.ViewIds)[1:max_id]:
    est.add_view(recon.View(v),  v, recon, 
        robust_kernel_width=2.,
        rolling_shutter=False,optimize_depth=False)
    print("added view: ",v)

# add gyro
telemetry_t_ns = telemetry_importer.telemetry["timestamps_ns"]
gyroscope = np.array(telemetry_importer.telemetry["gyroscope"])
accelerometer = np.array(telemetry_importer.telemetry["accelerometer"])

#est.add_gyroscope(gyroscope, telemetry_t_ns, 1.0)
#est.add_accelerometer(accelerometer, telemetry_t_ns, 1.0)

est.init_optimizer(nr_iter=10)

# reprojection error:
for vid in sorted(recon.ViewIds)[1:max_id]:

    view = recon.View(vid)
    T_c_w = est.get_camera_to_world_pose(torch.tensor(view.GetTimestamp()*1e9)).inverse()
    
    t_ids = view.TrackIds()

    errors = []
    for t_id in t_ids:
        pt4 = recon.Track(t_id).Point()
        p_in_cam = (T_c_w.to_matrix() @ torch.tensor(pt4)).squeeze()
        p_in_cam[:3] = p_in_cam[:3]/p_in_cam[2]
        p_in_img = torch.tensor(cam_matrix) @ p_in_cam[:3]
        err = (view.GetFeature(t_id).point - p_in_img.numpy()[:2])**2
        errors.append(err)
    
    repr_err = np.sqrt(np.mean(np.array(errors))) / len(t_ids)
    print("rmse reprojection error: ",repr_err)

est.forward()


# reprojection error:
for vid in sorted(recon.ViewIds)[1:max_id]:

    view = recon.View(vid)
    T_c_w = est.get_camera_to_world_pose(torch.tensor(view.GetTimestamp()*1e9)).inverse()
    
    t_ids = view.TrackIds()

    errors = []
    for t_id in t_ids:
        pt4 = recon.Track(t_id).Point()
        p_in_cam = (T_c_w.to_matrix() @ torch.tensor(pt4)).squeeze()
        p_in_cam[:3] = p_in_cam[:3]/p_in_cam[2]
        p_in_img = torch.tensor(cam_matrix) @ p_in_cam[:3]
        err = (view.GetFeature(t_id).point - p_in_img.numpy()[:2])**2
        errors.append(err)
    
    repr_err = np.sqrt(np.mean(np.array(errors))) / len(t_ids)
    print("rmse reprojection error: ",repr_err)