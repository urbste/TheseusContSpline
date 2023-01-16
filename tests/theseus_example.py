import torch
import matplotlib.pyplot as plt

torch.manual_seed(0)

def generate_data(num_points=100, a=1, b=0.5, noise_factor=0.01):
    # Generate data: 100 points sampled from the quadratic curve listed above
    data_x = torch.rand((1, num_points))
    noise = torch.randn((1, num_points)) * noise_factor
    data_y = a * data_x.square() + b + noise
    return data_x, data_y

def generate_learning_data(num_points, num_models):
    a, b = 3, 1
    data_batches = []
    for i in range(num_models):
        b = b + 2
        data = generate_data(num_points, a, b)
        data_batches.append(data)
    return data_batches

num_models = 10
data_batches = generate_learning_data(100, num_models)

fig, ax = plt.subplots()
for i in range(num_models):
    ax.scatter(data_batches[i][0], data_batches[i][1])
ax.set_xlabel('x')
ax.set_ylabel('y')


import theseus as th

data_x, data_y = data_batches[0]
x = th.Variable(data_x, name="x")
y = th.Variable(data_y, name="y")
a = th.Vector(1, name="a")
b = th.Vector(1, name="b")

# Note 'b' is the only optim_var, and 'a' is part of aux_vars
optim_vars = [b]
aux_vars = a, x, y

# updated error function reflects change in 'a'
def quad_error_fn2(optim_vars, aux_vars):
    [b] = optim_vars 
    a, x, y = aux_vars
    est = a.tensor * x.tensor.square() + b.tensor
    err = y.tensor - est
    return err

cost_function = th.AutoDiffCostFunction(
    optim_vars, quad_error_fn2, 100, aux_vars=aux_vars, name="quadratic_cost_fn"
)
objective = th.Objective()
objective.add(cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=50,
    step_size=0.5,
)
theseus_optim = th.TheseusLayer(optimizer)

# The value for Variable 'a' is optimized by PyTorch backpropagation
a_tensor = torch.nn.Parameter(torch.rand(1, 1))
model_optimizer = torch.optim.Adam([a_tensor], lr=0.1)


num_batches = len(data_batches)
num_epochs = 20

print(f"Initial a value: {a_tensor.item()}")

for epoch in range(num_epochs):
    epoch_loss = 0.
    epoch_b = []  # keep track of the current b values for each model in this epoch
    for i in range(num_batches):
        model_optimizer.zero_grad()
        data_x, data_y = data_batches[i]
        # Step 2.1: Create input dictionary for TheseusLayer, pass to forward function
        # The value for variable `a` is the updated `a_tensor` by Adam
        # Since we are always passing the same tensor, this update is technically redundant, 
        # we include it to explicitly illustrate where variable values come from.
        # An alternative way to do this (try it!)
        # would be to call `a.update(a_tensor)` before the learning loop starts
        # and just let Adam change its value under the hood (i.e., update only `x`, `y`, and `b`
        # inside the loop)
        theseus_inputs = {
            "a": a_tensor,
            "x": data_x,
            "y": data_y,
            "b": torch.ones((1, 1)),
        }
        updated_inputs, info = theseus_optim.forward(theseus_inputs)
        epoch_b.append(updated_inputs["b"].item())  # add the optimized "b" of the current batch to `epoch_b` list. 
                                                    # Note: here, we do not track the best solution, as we 
                                                    # backpropagate through the entire optimization sequence.
        # Step 2.2: Update objective function with updated inputs
        objective.update(updated_inputs)
        loss = cost_function.error().square().mean()
        # Step 2.3: PyTorch backpropagation
        loss.backward()
        model_optimizer.step()

        loss_value = loss.item()
        epoch_loss += loss_value
    print(f"Epoch: {epoch} Loss: {epoch_loss}")
    if epoch % 5 == 4:
        print(f" ---------------- Solutions at Epoch {epoch:02d} -------------- ")
        print(" a value:", a.tensor.item())
        print(" b values: ", epoch_b)
        print(f" ----------------------------------------------------- ")