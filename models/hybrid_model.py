import torch.nn as nn
from .classical_backbone import ResNetFeatureExtractor
from .quantum_layer import QuantumLayer

class HybridQuantumModel(nn.Module):
    def __init__(self, n_qubits=3, n_layers=2, feature_dim=8):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(output_dim=feature_dim)
        self.quantum_classifier = QuantumLayer(input_dim=feature_dim, n_qubits=n_qubits, n_layers=n_layers)

    def forward(self, x):
        classical_features = self.feature_extractor(x)
        return self.quantum_classifier(classical_features)

