import pennylane as qml

# Device =~= Computer
dev_gaussian = qml.device('default.gaussian', wires=1)


# Function =~= Quantum Circuit, converted into QNode on Device
@qml.qnode(dev_gaussian)
def mean_photon_gaussian(mag_alpha, phase_alpha, phi):
    qml.Displacement(mag_alpha, phase_alpha, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.expval(qml.NumberOperator(0))


# Squared Difference Cost
def cost(params):
    return (mean_photon_gaussian(params[0], params[1], params[2]) - 1) ** 2


# Initialise parameters (chosen to be arbitrarily small)
init_params = [0.015, 0.02, 0.0005]
print(cost(init_params))

# Initialise optimiser, step size and params
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 20
params = init_params

# Update params each step
for i in range(steps):
    params = opt.step(cost, params)
    print('Cost after step {}: {}'.format(i+1, cost(params)))

# Print optimizations
print('\nOptimised mag_alpha: {}'.format(params[0]))
print('Optimised phase_alpha: {}'.format(params[1]))
print('Optimised phi: {}'.format(params[2]))
