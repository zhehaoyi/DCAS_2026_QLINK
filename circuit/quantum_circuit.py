import tensorcircuit as tc
import torch


def quantum_res_circuit(num_qubits, circuit_depth, u_params, res_params, input_state, model, return_state=False):

    """
    :param num_qubits: number of qubits in the circuit
    :param circuit_depth: depth of the quantum circuit
    :param u_params: parameters for the quantum circuit block U (rx, ry, rz)
    :param res_params: parameters for the residual connections (rxx)
    :param input_state: input quantum statevector
    :param model: model type
    :return: quantum circuit probability or state with or without residual connections
    """

    qc = tc.Circuit(num_qubits, inputs = input_state)
    if model == "Vallina":
        data_idx = num_qubits  # idx of data qubits n
    else:
        data_idx = num_qubits - 1 # idx of data qubits (n - 1)
        control_idx = num_qubits - 1  # idx of control qubits the last qubit


    # if model is not Vallina, add residual connections
    if model != "Vallina":
        # first H on the control qubit to ensure the entanglement
        qc.h(control_idx)

        # initial residual connection via controll phase gate
        # if 
        for i in range(data_idx):
            if model == "Q-LINK(Fixed)":
                theta = torch.pi / 4
            else:
                theta = res_params[0, i]
            qc.rxx(i, control_idx, theta=theta)

    # place quantum circuit block U on data qubits
    for j in range(circuit_depth):
        for i in range(data_idx):
            qc.rz(i, theta=u_params[j, i, 0])
            qc.ry(i, theta=u_params[j, i, 1])
            qc.rx(i, theta=u_params[j, i, 2])
        
        for i in range(data_idx - 1):
            qc.cz(i, i + 1)
        
        if model != "Vallina":
            # add the residual back to data qubits through control not gate
            for i in range(data_idx):
                qc.cnot(control_idx, i)

            # new residual connection via controll phase gate before the last layer
            if j != circuit_depth - 1:
                for i in range(data_idx):
                    if model == "Q-LINK(Fixed)":
                        theta = torch.pi / 4
                    else:
                        theta = res_params[j + 1, i]
                    qc.rxx(i, control_idx, theta=theta)
    
    if return_state:
        return qc.state()
    else:
        return qc.probability()

