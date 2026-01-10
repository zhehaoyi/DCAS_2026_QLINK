import numpy as np
from scipy import stats
import tensorcircuit as tc
import torch
from circuit import quantum_res_circuit

class ComputeExpress:
    def __init__(self, num_bins, num_qubits, num_fidelity, depth, model):
        self.num_bins = num_bins
        self.num_qubits = num_qubits
        self.num_fidelity = num_fidelity
        self.depth = depth
        self.model = model
        
        # adjust num_qubits for no residual connection case
        self.data_n = num_qubits - 1 if model != "Vanilla" else num_qubits
        self.u_size = self.depth * self.data_n * 3
        self.res_size = (self.depth + 1) * self.data_n

        # compute Haar distribution
        # haar = (2^n - 1) * (1 - F)^(2^n - 2) n is the number of qubits, F is fidelity
        interval = 1 / self.num_bins
        bins_list = [interval * i for i in range(num_bins + 1)]
        dim = 2 ** self.num_qubits
        
        result = []
        for i in range(1, len(bins_list)):
            temp1 = 1 - np.power((1 - bins_list[i]), dim - 1)
            temp0 = 1 - np.power((1 - bins_list[i - 1]), dim - 1)
            result.append(temp1 - temp0)
        self.p_fidelity_Haar = np.array(result)

        self.vmap_cir_eval = tc.backend.jit(tc.backend.vmap(self.circuit_eval_state))

    def circuit_eval_state(self, param_flat):
        # reshape the param to u_params and res_params
        u_p = torch.reshape(param_flat[:self.u_size], (self.depth, self.data_n, 3))
        res_p = torch.reshape(param_flat[self.u_size:], (self.depth + 1, self.data_n))
        
        # call the quantum circuit to get the output state, the input_state is |0...0>
        return quantum_res_circuit(self.num_qubits, self.depth, u_p, res_p, None, self.model, return_state=True)

    def compute_express(self):
        # ramdomly generate parameters
        total_param_size = self.u_size + self.res_size
        param = torch.rand(self.num_fidelity * 2, total_param_size) * 2 * np.pi

        # get the output states
        output_state_all = self.vmap_cir_eval(param)
        
        # divide the output states into two groups to compute fidelity
        output_states1 = output_state_all[0:self.num_fidelity]
        output_states2 = output_state_all[self.num_fidelity:]

        # compute Fidelity: F = |<psi1|psi2>|^2
        # torch.conj(output_states2) conjugate
        fidelity = torch.abs(torch.sum(output_states1 * torch.conj(output_states2), dim=-1))**2
        fidelities = fidelity.detach().cpu().numpy()

        # distribution over bins
        bin_index = np.floor(fidelities * self.num_bins).clip(0, self.num_bins - 1).astype(int)
        num = [np.sum(bin_index == i) for i in range(self.num_bins)]
        p_fidelity = np.array(num) / np.sum(num)

        # compute KL divergence as expressibility
        express = stats.entropy(p_fidelity, self.p_fidelity_Haar)
        return express