from socket import SO_ACCEPTCONN
import theseus as th
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from cost_functions import accelerometer_error, so3_vel_error, so3_error, r3_error, r3_accl_error

from so3_spline import SO3Spline
from rd_spline import RDSpline
from time_util import S_TO_NS, calc_times

start_tns = 0.0 * S_TO_NS
end_tns = 55.0 * S_TO_NS
dt_ns_r3 = 0.5 * S_TO_NS
dt_ns_so3 = 0.5 * S_TO_NS

inv_dt_r3_s = S_TO_NS / dt_ns_r3
inv_dt_so3_s = S_TO_NS / dt_ns_so3

noise = 0.1
dt_points = 0.2 * S_TO_NS

DIM = 3
N = 4

num_knots_r3 = int((end_tns - start_tns) / dt_ns_r3) + N
num_knots_so3 = int((end_tns - start_tns) / dt_ns_so3) + N

so3_spline = SO3Spline(start_tns, end_tns, dt_ns=dt_ns_so3, N=N)
r3_spline = RDSpline(start_tns, end_tns, dt_ns=dt_ns_r3, dim=DIM, N=N)

# get a random trajectory and set knots
for i in range(num_knots_r3):
    rand_se3 = th.rand_se3(1)
    so3_spline.knots.append(th.SO3(tensor=rand_se3.rotation().tensor, name="so3_knot_"+str(i)))
    r3_spline.knots.append(th.Vector(tensor=rand_se3.translation().tensor, name="r3_knot_"+str(i)))


# sample 100 points on the spline
time_pos_meas = torch.arange(start_tns, end_tns, dt_points)
r3_measurements = []
r3_accl_measurements = []
rot_vel_measurements = []
so3_measurements = []

for i in range(len(time_pos_meas)):
    r3_pos = r3_spline.evaluate(time_pos_meas[i])
    r3_accl = r3_spline.evaluate(time_pos_meas[i], derivative=2)

    so3_rot = so3_spline.evaluate(time_pos_meas[i])
    so3_vel = so3_spline.velocityBody(time_pos_meas[i])

    r3_measurements.append(r3_pos+(np.random.randn(3)*noise).astype(np.float32))
    r3_accl_measurements.append(r3_accl+(np.random.randn(3)*noise).astype(np.float32))

    rot_vel_measurements.append(so3_vel+(np.random.randn(3)*noise).astype(np.float32))
    so3_measurements.append(so3_rot)

    # noise_quat = R.from_rotvec(np.random.randn(3)*noise).as_quat().astype(np.float32)

    # unit_quat = th.SO3(quaternion=torch.tensor(noise_quat[[3, 0, 1, 2]]))
    # rot_measurments_noisy.append(test_spline.evaluate(time_pos_meas[i]).compose(unit_quat))

    # noise_rot_vel = (test_spline.velocityBody(time_pos_meas[i]) + np.random.randn(3)*noise*0.1).float()
    # rot_vel_measurements.append(noise_rot_vel)

objective = th.Objective()
inv_dt_so3_s_ = th.Variable(tensor=torch.tensor(inv_dt_so3_s).unsqueeze(0), name="inv_dt_so3_s")
inv_dt_r3_s_ = th.Variable(tensor=torch.tensor(inv_dt_r3_s).unsqueeze(0), name="inv_dt_r3_s")


for k in range(len(r3_measurements)):

    u_r3, s_r3, suc = calc_times(time_pos_meas[k], r3_spline.start_time_ns, 
        r3_spline.dt_ns, len(r3_spline.knots), r3_spline.N)

    r3_meas = th.Variable(r3_measurements[k], name="r3_meas_"+str(k))
    r3_accl_meas = th.Variable(r3_accl_measurements[k].unsqueeze(0), name="r3_accl_meas_"+str(k))

    u_r3 = th.Variable(tensor=u_r3.unsqueeze(0), name="u_r3_"+str(k))

    aux_vars_r3 = [r3_meas, u_r3, inv_dt_r3_s_]
    aux_vars_r3_accl = [r3_accl_meas, u_r3, inv_dt_r3_s_]

    u_so3, s_so3, suc = calc_times(time_pos_meas[k], so3_spline.start_time_ns, 
        so3_spline.dt_ns, len(so3_spline.knots), so3_spline.N)

    rot_meas = th.Variable(so3_measurements[k].tensor, name="so3_meas_"+str(k))
    rot_vel_meas = th.Variable(rot_vel_measurements[k].unsqueeze(0), name="so3_vel_meas_"+str(k))

    u_so3 = th.Variable(tensor=u_so3.unsqueeze(0), name="u_so3_"+str(k))

    aux_vars_so3 = [rot_meas, u_so3, inv_dt_so3_s_]
    aux_vars_rot_vel = [rot_vel_meas, u_so3, inv_dt_so3_s_]

    aux_vars_accelerometer = [r3_accl_meas, u_r3, u_so3, inv_dt_r3_s_, inv_dt_so3_s_]


    optim_vars_r3, optim_vars_so3, optim_vars_se3 = [], [], []

    for i in range(r3_spline.N):
        optim_vars_r3.append(r3_spline.knots[i + s_r3])

    for i in range(so3_spline.N):
        optim_vars_so3.append(so3_spline.knots[i + s_so3])

    for i in range(so3_spline.N):
        quat = so3_spline.knots[i+s_so3].to_quaternion()
        pos = r3_spline.knots[i+s_r3].tensor
        optim_vars_se3.append(th.SE3(x_y_z_quaternion=torch.cat([pos, quat],1)))

    cost_function = th.AutoDiffCostFunction(
        optim_vars_se3, accelerometer_error, 3, aux_vars=aux_vars_accelerometer, name="accelerometer_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH)

    objective.add(cost_function)


    cost_function = th.AutoDiffCostFunction(
        optim_vars_so3, so3_error, 3, aux_vars=aux_vars_so3, name="so3_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH)

    objective.add(cost_function)

    cost_function = th.AutoDiffCostFunction(
        optim_vars_r3, r3_error, 3, aux_vars=aux_vars_r3, name="r3_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH)

    objective.add(cost_function)

    cost_function = th.AutoDiffCostFunction(
        optim_vars_so3, so3_vel_error, 3, aux_vars=aux_vars_so3, name="so3_vel_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH)

    objective.add(cost_function)

    cost_function = th.AutoDiffCostFunction(
        optim_vars_r3, r3_accl_error, 3, aux_vars=aux_vars_r3, name="r3_accl_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH)

    objective.add(cost_function)

# pose_prior_cost = th.Difference(
#     var=rot_measurments[0],
#     cost_weight=th.ScaleCostWeight(
#         torch.tensor(100, dtype=torch.float32)
#     ),
#     target=rot_measurments_noisy[0].copy(new_name=rot_measurments_noisy[0].name + "__PRIOR"),
# )
# objective.add(pose_prior_cost)

map_id_to_name = {}
theseus_inputs = {}
for i in range(len(r3_spline.knots)-3):
    knot = r3_spline.knots[i]
    theseus_inputs[knot.name] = r3_spline.knots[i].tensor
    map_id_to_name[i] = knot.name
for i in range(len(so3_spline.knots)-3):
    knot = so3_spline.knots[i]
    theseus_inputs[knot.name] = so3_spline.knots[i].tensor
    map_id_to_name[i] = knot.name

optimizer = th.LevenbergMarquardt(
    objective,
    max_iterations=15,
    step_size=0.5,
)
theseus_optim = th.TheseusLayer(optimizer)

with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})


# # set the optimized spline knots back to the spline
# for id in map_id_to_name:
#     knot_name = map_id_to_name[id]
#     if knot_name in updated_inputs:
#         test_spline.knots[id].update(updated_inputs[knot_name])

# pos_measurments_new = torch.zeros((time_pos_meas.shape[0], DIM)).float()
# for i in range(pos_measurments_new.shape[0]):
#     pos_measurments_new[i,:] = test_spline.evaluate(time_pos_meas[i])

# import matplotlib.pyplot as plt
# plt.plot(pos_measurments[:,0], 'r')
# plt.plot(pos_measurments_noisy[:,0], 'g*')
# plt.plot(pos_measurments_new[:,0], 'g')
# plt.show()
