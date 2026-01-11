# Q-LINK: Quantum Layerwise Information Residual Network via Messeager Qubit

This repository provides the reference implementation of **Q-LINK (Quantum Layerwise Information Residual Network via Messenger Qubit)**, a residual-like quantum circuit architecture designed to mitigate barren plateau effects in variational quantum algorithms by introducing a single messenger qubit.

Q-LINK enables layerwise residual information flow without circuit splitting or intermediate measurements, and improves both trainability and convergence efficiency.

The repository includes:
- **Q-LINK (Fixed)**: residual coupling gates are fixed.
- **Q-LINK (Adaptive)**: residual coupling gates are trainable.
- **Vanilla**: variational quantum circuit as a baseline (no messenger residual module).

The flowchatr as follow:

![image](qlink_flowchat\QLINK.png)
*Detailed architecture of the Q-LINK model. The circuit consists of a main variational circuit path acting on data qubits and a messenger qubit path that enables residual-like information exchange across layers. The interaction between the main path and the messenger path is illustrated schematically. Q-LINK includes a fixed with $R_{xx}$ gate parameters set to $\pi/4$, denoted as Q-LINK (Fixed) and an adaptive model with trainable $R_{xx}$ gate parameters denoted as Q-LINK (Adaptive)*

## Code Architecture
```bash
.
├── circuit/                # Circuit construction and quantum state management
│   ├── __init__.py
│   ├── quantum_circuit_depth.py
│   ├── quantum_circuit.py
│   └── quantum_state.py
├── metrics/                # Computing quantum metrics and gradients
│   ├── __init__.py
│   ├── compute_avg_loss_gradient.py
│   ├── compute_expressibility.py
│   └── cost_function.py
├── plots/                  # Data visualization
│   ├── __init__.py
│   ├── plot.py             # Plotting analysis results
│   └── replot.py           # Re-generation of figures using historical data
├── Q-LINK_results/           # Storage for all simulation outputs
│   ├── avg_stopping_iterations/ # Statistical data on convergence and different qubits
│   ├── data/                 # Data (CSV/JSON/NPZ) used for visualization
│   ├── loss_comparisons/     # Performance comparison plots across different qubits
│   ├── loss_landscapes/      # Visualizations of the cost function loss landscape
│   ├── replots/              # Figures re-generated from stored data
│   └── summaries/            # Summary reports and final performance metrics           
├── qlink_flowchart/        # Model architecture
│   └── QLINK.pdf           
├── .gitignore              
├── create_file.py          # Initialization for results directories
├── main.py                 
├── README.md               # Project documentation
└── requirements.txt        # List of Python dependencies
```
## Simulation Setup
All experiments are performed using stochastic gradient descent (SGD) with a learning rate of 0.1.  
- Number of qubits: 5–10 (including the messenger qubit)
- Circuit depth: $n^2\log (n)$
- Input states: random quantum states
- Number of repetitions: 5 for each model
- Convergence criterion: loss value < 0.001
- Maximum iteration: 1500

Metrics collected:
- Average stopping iteration
- Cost function loss landscape
- Gradient variance and Expressibility

## Running the Code
```bash
# first to install all the requirements
pip install requirements.txt

# then 
python main.py
```
This will:
- train all models for different numbers of qubits
- compute expressibility and gradient
- generate loss curves and loss landscapes
- save figures and  data under ```Q-LINK_results/```

## Authors
**Zhehao Yi** - *Algorithm Design, Implementation & Writing*  
**Rahul Bhadani** - *Project Guidance & Supervision*  
If you have any question, please reach out to **Zhehao Yi** at zhehao.yi@uah.edu

## License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
