"""
THE PARAMETER-SHIFT RULE

The gradient of the expectation value can be calculated by evaluating the same
variational quantum circuit, but with shifted parameter values
"""

import pennylane as qml
from pennylane import numpy as np
import timeit

# Set random seed for cross referencing
np.random.seed(42)

# Initialise device
dev = qml.device('default.qubit', wires=3)


# Initialise QNode connection circuit to device
@qml.qnode(dev, diff_method='parameter-shift')
def circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')

    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)

    qml.broadcast(qml.CNOT, wires=[0, 1, 2], pattern='ring')
    return qml.expval(qml.PauliY(0) @ qml.PauliZ(2))


# Parameter shift term by term
def parameter_shift_term(qnode, params, i):
    shifted = params.copy()
    shifted[i] += np.pi/2
    forward = qnode(shifted)

    shifted[i] -= np.pi
    backward = qnode(shifted)

    return 0.5 * (forward - backward)


# Parameter shift
def parameter_shift(qnode, params):
    gradients = np.zeros([len(params)])

    for i in range(len(params)):
        gradients[i] = parameter_shift_term(qnode, params, i)

    return gradients


# Gradient wrt first term
params = np.random.random([6])
print('Parameters: ', params)
print('Expectation Value: ', circuit(params))

# Draw the circuit
print(circuit.draw())

print(parameter_shift_term(circuit, params, 0))

# Gradient wrt all terms
print(parameter_shift(circuit, params))

# Compare to Pennylane built in parameter shift function
grad_function = qml.grad(circuit)
print(grad_function(params)[0])


"""
BENCHMARKING

Example with significantly larger number of parameters
"""

dev = qml.device('default.qubit', wires=4)


@qml.qnode(dev, diff_method='parameter-shift', mutable=False)
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
print(params.size)
print(circuit(params))

# Timing forward pass of circuit
reps = 3
num = 10
times = timeit.repeat('circuit(params)', globals=globals(), number=num, repeat=reps)
forward_time = min(times) / num
print('Forward pass (best of {}): {} sec per loop'.format(reps, forward_time))

# Time for full gradient vector
# grad_fn = qml.grad(circuit)
# times = timeit.repeat('grad_fn(params)', globals=globals(), number=num, repeat=reps)
# backward_time = min(times) / num
# print('Gradient computation (best of {}): {} sec per loop'.format(reps, backward_time))

# Time to compute quantum gradients (above is ~5s/loop)
print(2 * forward_time * params.size)
