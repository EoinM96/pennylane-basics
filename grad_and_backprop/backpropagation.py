import tensorflow as tf
import pennylane as qml
import timeit

dev = qml.device('default.qubit.tf', wires=4)


@qml.qnode(dev, diff_method='backprop', interface='tf')
def circuit(params):
    qml.templates.StronglyEntanglingLayers(params, wires=[0, 1, 2, 3])
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1) @ qml.PauliZ(2) @ qml.PauliZ(3))


params = qml.init.strong_ent_layers_normal(n_wires=4, n_layers=15)
params = tf.Variable(params)
print(circuit(params))
