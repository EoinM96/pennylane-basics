import pennylane as qml
from pennylane import numpy as np

# Define device using SF's Fock backend
dev_fock = qml.device('strawberryfields.fock', wires=2, cutoff_dim=2)


# QNode linking function to device
@qml.qnode(dev_fock)
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))


# Optimisation
def cost(params):
    return -photon_redirection(params)


# Initial parameters
init_params = np.array([0.01, 0.01])
print(cost(init_params))

# Zero gradient confirmed verification
dphoton_redirection = qml.grad(photon_redirection, argnum=0)
print(dphoton_redirection([0, 0]))

# Initialise optimiser
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100
params = init_params

# Update params each step
for i in range(steps):
    params = opt.step(cost, params)

    if (i+1) % 5 == 0:
        print('Cost after step {}: {}'.format(i+1, cost(params)))

# Print output
print('Optimsed rotation angles; {}'.format(params))
