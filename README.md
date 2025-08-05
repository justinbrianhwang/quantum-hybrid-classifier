# Quantum-Classical Hybrid Image Classifier

A PyTorch + PennyLane based **Quantum-Classical Hybrid Image Classifier**  
that supports both **classical deep learning models** and a **quantum-classical hybrid model**  
for image classification tasks.

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/17cd38f9-4957-435f-b89e-a35709319d88" />


This project is designed as a **general image classification framework**,  
so you can either use built-in datasets (e.g., CIFAR-10) or **your own custom dataset**.

---


## Project Structure

```
quantum-hybrid-classifier/
│
├─ data/ # Place your dataset here (empty by default)
│ ├─ train/
│ │ ├─ class_0/
│ │ └─ class_1/
│ ├─ val/
│ │ ├─ class_0/
│ │ └─ class_1/
│ └─ test/
│ ├─ class_0/
│ └─ class_1/
│
├─ datasets/
│ ├─ init.py
│ ├─ custom_dataset.py # For custom datasets (in data/)
│ └─ cifar10.py # Example for CIFAR-10
│
├─ models/
│ ├─ classical_backbone.py # ResNet feature extractor for hybrid model
│ ├─ classical_baseline.py # Classical & compressed ResNet models
│ ├─ hybrid_model.py # Quantum-classical hybrid model
│ └─ quantum_layer.py # PennyLane-based quantum circuit
│
├─ trainers/
│ └─ trainer.py # Training logic for all models
│
├─ utils/
│ ├─ evaluator.py # Evaluation metrics & confusion matrix
│ ├─ visualization.py # Training curve plotting
│ └─ common.py # Utilities (e.g., random seed)
│
├─ main.py # Unified training & evaluation script
├─ requirements.txt
├─ LICENSE (MIT)
└─ README.md
```

---

## 🚀 Features

- **Classical & Hybrid Models**
  - `classical` : Standard ResNet-18
  - `compressed` : Lightweight ResNet variant with fewer parameters
  - `hybrid` : ResNet-18 feature extractor + Quantum Layer

- **Quantum Integration**
  - PennyLane + Qiskit backend
  - Supports **simulator mode** and **IBM QPU mode** (requires token)

- **Flexible Dataset Handling**
  - Built-in support: **CIFAR-10**, **MNIST**
  - **Custom datasets** via simple folder structure

- **Evaluation**
  - Accuracy, Precision, Recall, F1-score, AUC-ROC
  - Confusion matrix & training curves auto-generated

---

## 📦 Installation

```bash
git clone https://github.com/justinbrianhwang/quantum-hybrid-classifier.git
cd quantum-hybrid-classifier
pip install -r requirements.txt
```

## Usage
```bash
python main.py --dataset cifar10 --download --model hybrid --epochs 30
```

## Train with a custom dataset
1. Prepare your dataset under `data/`:
```
data/
  train/
      class_0/
      class_1/
  val/
      class_0/
      class_1/
  test/
      class_0/
      class_1/
```
2. Run training
```bash
python main.py --dataset custom --data_dir ./data --model classical --epochs 20
```

## ⚡ Arguments

| Argument        | Description                                  | Default |
|-----------------|----------------------------------------------|--------|
| `--dataset`     | `custom` / `cifar10` / `mnist`               | custom |
| `--data_dir`    | Path to dataset                              | ./data |
| `--download`    | Auto-download if dataset is supported        | False  |
| `--model`       | `hybrid` / `classical` / `compressed`        | hybrid |
| `--epochs`      | Training epochs                              | 20     |
| `--batch_size`  | Batch size                                   | 32     |
| `--n_qubits`    | Number of qubits (hybrid only)               | 3      |
| `--n_layers`    | Quantum circuit layers                       | 2      |
| `--mode`        | 1: simulator / 2: IBM QPU                    | 1      |
| `--lr`          | Learning rate                                | 1e-4   |


## 🧩 Modifying for Your Own Dataset
- If you don't want to use CIFAR-10, simply:
    1. Place your dataset under data/ using the same folder structure.
    2. Use --dataset custom in your training command.
    3. If your dataset has more than 2 classes, the code automatically adapts.
- If you need special preprocessing:
    - Edit datasets/`custom_dataset.py` to implement your transform logic.

## Example Outputs
- `training_curves.png` : Training & validation loss/accuracy
- `confusion_matrix.png` : Test set confusion matrix
- `best_hybrid_model.pth` : Saved model weights

## 📄 License
This project is licensed under the MIT License.
Feel free to modify and use it in your own projects.

## 🙌 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [PennyLane](https://pennylane.ai/)
- [Qiskit](https://qiskit.org/)
- [TorchVision](https://pytorch.org/vision/stable/index.html)



