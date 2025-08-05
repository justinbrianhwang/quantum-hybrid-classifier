import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
import time

class Trainer:
    def __init__(self, model, device='cuda', model_type='hybrid', lr=1e-4):
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss() if model_type != 'classical' else nn.CrossEntropyLoss()

        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def train(self, train_loader, val_loader, epochs=20):
        print(f"=== Training {self.model_type.upper()} model for {epochs} epochs ===")
        best_val_acc = 0.0
        for epoch in range(epochs):
            start_time = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._validate_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')

            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
                  f"Time: {time.time()-start_time:.1f}s")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

    def _train_epoch(self, train_loader):
        self.model.train()
        total_loss, total_correct, total = 0.0, 0, 0
        for data, target in train_loader:
            data = data.to(self.device).float()
            target = target.to(self.device)
            if self.model_type != 'classical':
                target = target.float().unsqueeze(1)

            self.optimizer.zero_grad()
            output = self.model(data)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            if self.model_type == 'classical':
                predicted = output.argmax(dim=1)
                total_correct += (predicted == target).sum().item()
            else:
                predicted = (output > 0.5).float()
                total_correct += (predicted == target).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), 100. * total_correct / total

    def _validate_epoch(self, val_loader):
        self.model.eval()
        total_loss, total_correct, total = 0.0, 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.device).float()
                target = target.to(self.device)
                if self.model_type != 'classical':
                    target = target.float().unsqueeze(1)

                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                if self.model_type == 'classical':
                    predicted = output.argmax(dim=1)
                    total_correct += (predicted == target).sum().item()
                else:
                    predicted = (output > 0.5).float()
                    total_correct += (predicted == target).sum().item()
                total += target.size(0)

        return total_loss / len(val_loader), 100. * total_correct / total

