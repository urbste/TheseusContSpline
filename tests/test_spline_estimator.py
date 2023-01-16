# created by Steffen Urban, 11/2022
import torch
import SplineEstimator.spline.time_util as time_util
import SplineEstimator.residuals.camera as cam_res

import theseus as th
import pytheia as pt
import numpy as np
from SplineEstimator.SplineEstimator3D import SplineEstimator3D
from scipy.spatial.transform import Rotation as R
import json

def read_json(json_file):
    with open(json_file) as f:
        data = json.load(f)
    return data

def load_imu_data(sensor_data, cam_t_ns):
    accl = sensor_data["linear_acceleration"]
    gyro = sensor_data["angular_velocity"]
    accl_m = np.array([a for a in accl.values()])
    gyro_m = np.array([[g[0],g[1],g[2]] for g in gyro.values()])*RHO
    imu_times_ns = np.array([int(t) for t in accl.keys()])
    # grab only relevant IMU data
    valid_imu_samples = np.where(np.logical_and(imu_times_ns > cam_t_ns[0], imu_times_ns < cam_t_ns[-1]))[0]
    imu_times_ns = imu_times_ns[valid_imu_samples]
    accl_m = accl_m[valid_imu_samples,:]
    gyro_m = gyro_m[valid_imu_samples,:]

    return accl_m, gyro_m, imu_times_ns

def load_imu_cam_calib(sensor_data):
    T_c_i = np.eye(4)
    R_c_i = R.from_quat(sensor_data["imu_to_cam_quaternion_xyzw"]).as_matrix()
    t_c_i = np.array(sensor_data["imu_to_cam_translation"])
    T_c_i[:3, :3] = R_c_i
    T_c_i[:3, 3] = t_c_i
    T_i_c = np.linalg.inv(T_c_i)

    return T_i_c


class IMUWeightRegressor(torch.nn.Module):
    def __init__(self, in_dim=6) -> None:
        super().__init__()

        self.layer1 = torch.nn.Conv1d(in_channels=in_dim, out_channels=16, kernel_size=5)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
        self.act2 = torch.nn.ReLU()
        self.lin1 = torch.nn.Linear(16, 2, bias=True)
        self.act3 = torch.nn.ReLU()
    def forward(self, imu_data):
        x = self.layer1(imu_data)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = x.mean(-1)

        x = self.lin1(x)
        x = self.act3(x) + 1.0
        return x

# define IMU weighting factors that we want to learn
imu_weight_cnn = IMUWeightRegressor(6)
outer_optimizer = torch.optim.AdamW(
    imu_weight_cnn.parameters(), lr=0.1)

device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"
RHO = np.pi / 180.0 

# load sfm data
recon = pt.io.ReadReconstruction("./test_data/pytheia_recon.recon")[1]
cam_matrix = recon.View(0).Camera().GetCalibrationMatrix()

sensor_data = read_json("./test_data/data.svo_sensor_data.json")
left_cam = sensor_data["camera"]["left"]

T_i_c = load_imu_cam_calib(sensor_data)

# setup a VI Spline and initialize it with SfM data
sorted_vids = sorted(recon.ViewIds)
max_id = 20
max_ts = recon.View(max_id).GetTimestamp()
spline_degree = 4
r3_dt_ns = 0.1 * time_util.S_TO_NS
so3_dt_ns = 0.1 * time_util.S_TO_NS

est = SplineEstimator3D(spline_degree, so3_dt_ns, r3_dt_ns, 
    T_i_c, cam_matrix, device)
est.init_spline_with_vision(recon, max_time_s=max_ts)

cam_t_ns = []
for vid in sorted(recon.ViewIds)[1:]:
    est.add_view(vid, recon, 
        robust_kernel_width=5.,
        rolling_shutter=True,optimize_depth=False)
    cam_t_ns.append(int(recon.View(vid).GetTimestamp()*time_util.S_TO_NS))
    if vid > max_id:
        break

# load IMU and get gravity direction of local map (pattern)
accl_m, gyro_m, imu_times_ns = load_imu_data(sensor_data, cam_t_ns)
R_w_c = recon.View(0).Camera().GetOrientationAsRotationMatrix().T 
g_world = R_w_c @ T_i_c[:3,:3].T @ accl_m[0,:]

# add gyro and accel data
est.add_gyroscope(accl_m, imu_times_ns, torch.tensor([[1.0]]))
est.add_accelerometer(gyro_m, imu_times_ns, torch.tensor([[1.0]]))
est.set_gravity(g_world)

est.init_optimizer(maxiter=2)

for epoch in range(10):
    outer_optimizer.zero_grad()

    imu_set = torch.cat([torch.tensor(accl_m).T, torch.tensor(gyro_m).T],0).unsqueeze(0)
    imu_weights = imu_weight_cnn(imu_set)

    sol, error = est.forward(
        {"gyro_cost_weight": imu_weights[[0],0].unsqueeze(0), 
         "accl_cost_weight": imu_weights[[0],1].unsqueeze(0)})

    # loss of entire problem
    loss = est.objective.error_squared_norm()
    loss.backward()
    outer_optimizer.step()

    print(imu_weights)