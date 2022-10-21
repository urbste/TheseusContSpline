from matplotlib import test
from numpy import vectorize
import theseus as th
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
from cost_functions import spline_rot_error

from so3_spline import SO3Spline
from spline_helper import SplineHelper
from time_util import S_TO_NS, calc_times

start_tns = 0.0 * S_TO_NS
end_tns = 5.0 * S_TO_NS
dt_ns = 0.5 * S_TO_NS
inv_dt_s = S_TO_NS / dt_ns
noise = 0.01
dt_points = 0.2 * S_TO_NS
DIM = 3
N = 4

test_spline = SO3Spline(start_tns, end_tns, dt_ns=dt_ns, dim=DIM, N=N)
test_spline.genRandomTrajectory()
s_knots = test_spline.knots

# sample 100 points on the spline
time_pos_meas = torch.arange(start_tns, end_tns, dt_points)
rot_measurments = []
rot_measurments_noisy = []

for i in range(len(time_pos_meas)):
    rot_measurments.append(test_spline.evaluate(time_pos_meas[i]))
    noise_quat = R.from_rotvec(np.random.randn(3)*noise).as_quat().astype(np.float32)

    unit_quat = th.SO3(quaternion=torch.tensor(noise_quat[[3, 0, 1, 2]]))
    rot_measurments_noisy.append(test_spline.evaluate(time_pos_meas[i]).compose(unit_quat))

# import matplotlib.pyplot as plt
# plt.plot(rot_measurments[:,0], 'r')
# plt.plot(rot_measurments_noisy[:,0], 'g*')
# plt.show()

objective = th.Objective()
inv_dt_s_ = th.Variable(tensor=torch.tensor(inv_dt_s).unsqueeze(0), name="inv_dt_s")

theseus_inputs = {}

for k, rot_m in enumerate(rot_measurments_noisy):

    u, s, suc = calc_times(time_pos_meas[k], test_spline.start_time_ns, 
        test_spline.dt_ns, len(test_spline.knots), test_spline.N)

    measurement = th.Variable(rot_m.tensor, name="rot_meas_"+str(k))

    u = th.Variable(tensor=u.unsqueeze(0), name="u"+str(k))

    aux_vars = [measurement, u, inv_dt_s_]
    optim_vars = []

    for i in range(test_spline.N):
        optim_vars.append(test_spline.knots[i + s])

    cost_function = th.AutoDiffCostFunction(
        optim_vars, spline_rot_error, DIM, aux_vars=aux_vars, name="pos_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH
    )
    objective.add(cost_function)

pose_prior_cost = th.Difference(
    var=rot_measurments[0],
    cost_weight=th.ScaleCostWeight(
        torch.tensor(100, dtype=torch.float32)
    ),
    target=rot_measurments_noisy[0].copy(new_name=rot_measurments_noisy[0].name + "__PRIOR"),
)
objective.add(pose_prior_cost)

map_id_to_name = {}
for i in range(len(test_spline.knots)-3):
    knot = test_spline.knots[i]
    theseus_inputs[knot.name] = test_spline.knots[i].tensor
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
