import argparse
import torch
from datasets import get_dataset_and_loaders
from models.hybrid_model import HybridQuantumModel
from models.classical_baseline import ClassicalResNet, CompressedClassical
from trainers.trainer import Trainer
from utils.evaluator import ModelEvaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Quantum-Classical Hybrid Image Classifier")
    parser.add_argument('--dataset', type=str, default='custom', help='Dataset name: custom | cifar10 | mnist')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--download', action='store_true', help='Download dataset if supported')
    parser.add_argument('--model', type=str, default='hybrid', help='Model type: hybrid | classical | compressed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_qubits', type=int, default=3)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--mode', type=int, default=1, help='1: simulator, 2: IBM QPU')
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Dataset & DataLoader
    train_loader, val_loader, test_loader, num_classes = get_dataset_and_loaders(
        args.dataset, args.data_dir, args.batch_size, args.download
    )

    # Model selection
    if args.model == 'hybrid':
        model = HybridQuantumModel(n_qubits=args.n_qubits, n_layers=args.n_layers, feature_dim=8)
    elif args.model == 'classical':
        model = ClassicalResNet(num_classes=num_classes)
    elif args.model == 'compressed':
        model = CompressedClassical()
    else:
        raise ValueError("Invalid model type")

    # Training & Evaluation
    trainer = Trainer(model, device=device, model_type=args.model, lr=args.lr)
    trainer.train(train_loader, val_loader, epochs=args.epochs)

    evaluator = ModelEvaluator(model, device=device)
    results = evaluator.evaluate(test_loader)
    print(results)

if __name__ == '__main__':
    main()

