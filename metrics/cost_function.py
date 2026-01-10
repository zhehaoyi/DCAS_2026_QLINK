import tensorcircuit as tc

def cost_function(probs, num_qubits, model):
    s = 0
    if model != "Vanilla":
        num_qubits = num_qubits - 1  # exclude the control qubit
    # iterate over n-1 quantum bit (excluding the control qubit)
    for i in range(num_qubits):
        # for each quantum bit, calculate the probability of it being 0
        # iterate over all possible states (2^num_qubits states in total)
        for j in range(0, 2 ** (num_qubits)):
            # convert the state index `j` to a binary string and check if the i-th bit is 0
            if format(j, f'0{num_qubits}b')[i] == '0':
                # accumulate the probability of the state where the i-th bit is 0
                s += probs[j]

    # compute and return the value of the cost function
    # return 1 minus the average probability of all quantum bits being 0
    return 1 - s / (num_qubits)