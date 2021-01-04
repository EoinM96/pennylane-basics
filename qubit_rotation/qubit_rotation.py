import pennylane as qml
from pennylane import numpy as np

# Quantum Device is any computational object that can apply quantum operations and return a measurement value
# Wires = Num of subsystems to initialise device with. We require a single qubit, ie wires=1
dev1 = qml.device('default.qubit', wires=1)


# Quantum function (gate) evaluated by QNode (abstract encapsulation of q.func described by a quantum circuit
@qml.qnode(dev1)
def circuit(params):
    """
    RX and RY are rotational gates (see notes)

    :param params: x,y args
    :return: Expected value w/ Pauli-Z gate
    """
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)

    return qml.expval(qml.PauliZ(0))


print(circuit([0.54, 0.12]))

# Grad of the func 'circuit' (within QNode) can be evaluated w/ same device 'dev1'
dcircuit = qml.grad(circuit, argnum=0)  # Returns func representing grad of circuit
print(dcircuit([0.54, 0.12]))


# We can optimise params s.t. qubit in |0> is rotated to be in state |1>
# Equiv to measuring a Pauli-Z expectation value of -1
# We define a cost function (here, equal to the output of the QNode)
def cost(x):
    return circuit(x)


# Given initial values, print cost
init_params = np.array(([0.011, 0.012]))
print(cost(init_params))

# Initialise optimiser
opt = qml.GradientDescentOptimizer(stepsize=0.4)
steps = 100
params = init_params

# Update circuit params each iteration
for i in range(steps):
    params = opt.step(cost, params)

    if (i+1) % 5 == 0:
        print('Cost after step {}: {}'.format(i+1, cost(params)))

# Print optimised rotational angles
print('Optimised rotation angles: {}'.format(params))
