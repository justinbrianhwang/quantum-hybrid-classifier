import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import os

class QuantumCircuit:
    def __init__(self, n_qubits=3, n_layers=2, mode=1, backend_sim='qasm_simulator', backend_qpu='ibmq_qasm_simulator', shots=1024):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.shots = shots

        if mode == 1:
            print("🔹 Simulator mode")
            self.dev = qml.device("default.qubit", wires=n_qubits, shots=shots)
        elif mode == 2:
            print("🔹 IBM QPU mode")
            token = os.getenv("QISKIT_IBM_API_TOKEN")
            if token is None:
                raise ValueError("Set QISKIT_IBM_API_TOKEN for QPU mode.")
            self.dev = qml.device("qiskit.runtime", wires=n_qubits, backend=backend_qpu, shots=shots)
        else:
            raise ValueError("mode must be 1(sim) or 2(QPU)")

        @qml.qnode(self.dev, interface='torch')
        def circuit(inputs, weights):
            for i in range(self.n_qubits):
                if i < len(inputs):
                    qml.RY(np.pi * inputs[i].item(), wires=i)
            for layer in range(self.n_layers):
                for i in range(self.n_qubits):
                    qml.RY(weights[layer, i], wires=i)
                for i in range(self.n_qubits):
                    qml.CNOT(wires=[i, (i+1) % self.n_qubits])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def forward(self, inputs, weights):
        return self.circuit(inputs.cpu(), weights.cpu())

class QuantumLayer(nn.Module):
    def __init__(self, input_dim=8, n_qubits=3, n_layers=2, mode=1, shots=1024):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.quantum_circuit = QuantumCircuit(n_qubits, n_layers, mode=mode, shots=shots)
        self.quantum_weights = nn.Parameter(torch.randn(n_layers, n_qubits) * 0.1)

    def forward(self, x):
        batch_size = x.shape[0]
        outputs = []
        for i in range(batch_size):
            quantum_output = self.quantum_circuit.forward(x[i], self.quantum_weights)
            outputs.append(quantum_output)
        outputs = torch.stack(outputs)
        probabilities = (outputs + 1) / 2
        return probabilities.unsqueeze(1)

