import numpy as np
from qiskit.quantum_info import random_statevector


def create_random_state(num_qubits, model):
    """
    Docstring for create_random_state
    
    :param num_qubits: number of qubits
    :param model: model type ("Q-LINK(Fixed)", "Q-LINK(Adaptive)", or "Vanilla")
    :return: random quantum statevector
    """
    # get a random input quantum state
    data_state = random_statevector(2 ** (num_qubits))
    if model == "Q-LINK(Fixed)" or model == "Q-LINK(Adaptive)":
        control_state = np.array([1.0, 0.0], dtype=np.complex128)
        input_state = np.kron(data_state, control_state)
    else:
         input_state = data_state
    return input_state