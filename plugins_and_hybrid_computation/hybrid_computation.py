import pennylane as qml
from pennylane import numpy as np

# Define devices
dev_qubit = qml.device('default.qubit', wires=1)
dev_fock = qml.device('strawberryfields.fock', wires=2, cutoff_dim=10)


# QNode's linking functions to devices
@qml.qnode(dev_qubit)
def qubit_rotation(phi1, phi2):
    qml.RX(phi1, wires=0)
    qml.RY(phi2, wires=0)
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev_fock)
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0, 1])
    return qml.expval(qml.NumberOperator(1))


# Error function
def squared_difference(x, y):
    return np.abs(x - y) ** 2

# Cost function
def cost(params, phi1=0.5, phi2=0.1):
    qubit_result = qubit_rotation(phi1, phi2)
    photon_result = photon_redirection(params)
    return squared_difference(qubit_result, photon_result)


# Initialise optimiser
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100
params = np.array([0.01, 0.01])

# Update params each step
for i in range(steps):
    params = opt.step(cost, params)

    if (i+1) % 5 == 0:
        print('Cost after step {}: {}'.format(i+1, cost(params)))

# Print output
print('\nOptimised rotation angles: {}\n'.format(params))

# Print optimisations
result = [1.20671364, 0.01]
print(photon_redirection(result))
print(qubit_rotation(0.5, 0.1))
