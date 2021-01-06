"""
BACKPROPAGATION

An alternative to the parameter shift rule is reverse-mode autodiffereniation. This method uses a single
forward pass of the differentiable function to compute the gradient of all variables, at the expense of
increased memory usage
"""

import tensorflow as tf
import pennylane as qml
import timeit

# Initialise device
dev = qml.device('default.qubit.tf', wires=4)


# Initialise QNode connection circuit to device
@qml.qnode(dev, diff_method='backprop', interface='tf')
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


# Initialising params
params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
params = tf.Variable(params)
print(circuit(params))

# Timing forward pass of circuit
reps = 3
num = 10
times = timeit.repeat('circuit(params)', globals=globals(), number=num, repeat=reps)
forward_time = min(times) / num
print('Forward pass (best of {}): {}s / loop'.format(reps, forward_time))

# Timing backward pass of circuit
with tf.GradientTape(persistent=True) as tape:
    res = circuit(params)

times = timeit.repeat('tape.gradient(res, params)', globals=globals(), number=num, repeat=reps)
backward_time = min(times) / num
print('Backward pass (best of {}): {}s / loop'.format(reps, backward_time))
