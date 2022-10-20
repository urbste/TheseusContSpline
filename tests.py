from matplotlib import test
from numpy import vectorize
import theseus as th
import torch

from cost_functions import spline_position_error

from rd_spline import RDSpline
from spline_helper import SplineHelper
from time_util import S_TO_NS, calc_times

start_tns = 0.0 * S_TO_NS
end_tns = 200.0 * S_TO_NS
dt_ns = 0.2 * S_TO_NS
inv_dt_s = S_TO_NS / dt_ns
noise = 0.1
dt_points = 0.1 * S_TO_NS
DIM = 2
N = 4

test_spline = RDSpline(start_tns, end_tns, dt_ns=dt_ns, dim=DIM, N=N)
test_spline.genRandomTrajectory()
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([0.,0.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(0)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([1.,1.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(1)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([2.,2.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(2)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([3.,3.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(3)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([4.,4.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(4)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([5.,5.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(5)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([6.,6.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(6)))
# test_spline.knots.append(
#                 th.Vector(tensor=torch.tensor([7.,7.]).unsqueeze(0), 
#                     name="r"+str(test_spline.DIM)+"_knot_"+str(7)))
s_knots = test_spline.knots

# sample 100 points on the spline
time_pos_meas = torch.arange(start_tns, end_tns, dt_points)
pos_measurments = torch.zeros((time_pos_meas.shape[0], DIM)).float()
pos_measurments_noisy = torch.zeros((time_pos_meas.shape[0], DIM)).float()
for i in range(pos_measurments.shape[0]):
    pos_measurments[i,:] = test_spline.evaluate(time_pos_meas[i])
    pos_measurments_noisy[i,:] = pos_measurments[i,:] + torch.randn(2)*noise

import matplotlib.pyplot as plt
plt.plot(pos_measurments[:,0], 'r')
plt.plot(pos_measurments_noisy[:,0], 'g*')
plt.show()

objective = th.Objective()
inv_dt_s_ = th.Variable(tensor=torch.tensor(inv_dt_s).unsqueeze(0), name="inv_dt_s")

theseus_inputs = {}

for k in range(pos_measurments_noisy.shape[0]):

    u, s, suc = calc_times(time_pos_meas[k], test_spline.start_time_ns, 
        test_spline.dt_ns, len(test_spline.knots), test_spline.N)

    measurement = th.Variable(tensor=pos_measurments_noisy[k,:].unsqueeze(0), name="pos_meas_"+str(k))

    u = th.Variable(tensor=u.unsqueeze(0), name="u"+str(k))

    aux_vars = [measurement, u, inv_dt_s_]
    optim_vars = []

    for i in range(test_spline.N):
        optim_vars.append(test_spline.knots[i + s])

    cost_function = th.AutoDiffCostFunction(
        optim_vars, spline_position_error, DIM, aux_vars=aux_vars, name="pos_cost_"+str(k), 
        autograd_vectorize=True, autograd_mode=th.AutogradMode.LOOP_BATCH
    )
    objective.add(cost_function)

map_id_to_name = {}
for i in range(len(test_spline.knots)-3):
    knot = test_spline.knots[i]
    theseus_inputs[knot.name] = test_spline.knots[i].tensor
    map_id_to_name[i] = knot.name

optimizer = th.GaussNewton(
    objective,
    max_iterations=15,
    step_size=0.5,
)
theseus_optim = th.TheseusLayer(optimizer)

with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose":True})


# set the optimized spline knots back to the spline
for id in map_id_to_name:
    knot_name = map_id_to_name[id]
    if knot_name in updated_inputs:
        test_spline.knots[id].update(updated_inputs[knot_name])

pos_measurments_new = torch.zeros((time_pos_meas.shape[0], DIM)).float()
for i in range(pos_measurments_new.shape[0]):
    pos_measurments_new[i,:] = test_spline.evaluate(time_pos_meas[i])

import matplotlib.pyplot as plt
plt.plot(pos_measurments[:,0], 'r')
plt.plot(pos_measurments_noisy[:,0], 'g*')
plt.plot(pos_measurments_new[:,0], 'g')
plt.show()
