import numpy as np


def circuit_depth(num_qubits):
    """
    Docstring for circuit_depth
    
    :param num_qubits: number of qubits in the circuit
    :return: depth of the quantum circuit
    """
    # For this example, we define the circuit depth as ceil(n^2 * log(n)) the number of qubits
    return int(np.ceil(num_qubits ** 2 * np.log(num_qubits))) 