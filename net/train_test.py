import torch as t
from config import DEVICE

class TrainerTester:
    def __init__(self, model: t.nn.Module):
        self.device = DEVICE
        self.model = model.to(self.device)
    
    def train_test(self, criterion: t.nn.Module,
              train_dataloader: t.utils.data.DataLoader,
              epochs: int, lr: float, optimizer_class: t.optim.Optimizer,
              test_dataloader: t.utils.data.DataLoader,
              **optimizer_kwargs) -> dict:
        
        optimizer = optimizer_class(
            params = self.model.parameters(),
            lr = lr,
            **optimizer_kwargs
        )
        
        history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        
        for _ in range(epochs):
            
            self.model.train()
            epoch_loss = 0
            epoch_acc = 0
            
            for batch_x, batch_y in train_dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                y_pred = self.model(batch_x)
                loss = criterion(y_pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += self._compute_accuracy(y_pred, batch_y)
            
            avg_loss = epoch_loss / len(train_dataloader)
            avg_acc = epoch_acc / len(train_dataloader)
            
            history["train_loss"].append(avg_loss)
            history["train_acc"].append(avg_acc)
            
            if test_dataloader:
                self.model.eval()
                test_loss = 0
                test_acc = 0
                
                with t.no_grad():
                    for batch_x, batch_y in test_dataloader:
                        batch_x = batch_x.to(self.device)
                        batch_y = batch_y.to(self.device)
                        
                        y_pred = self.model(batch_x)
                        loss = criterion(y_pred, batch_y)
                        
                        test_loss += loss.item()
                        test_acc += self._compute_accuracy(y_pred, batch_y)
                
                avg_test_loss = test_loss / len(test_dataloader)
                avg_test_acc = test_acc / len(test_dataloader)
                
                history["test_loss"].append(avg_test_loss)
                history["test_acc"].append(avg_test_acc)
        
        return history
    
    def _compute_accuracy(self, y_pred: t.Tensor, y_true: t.Tensor) -> float:
        _, predicted = t.max(y_pred, 1)
        correct = (predicted == y_true).sum().item()
        return correct / len(y_true)
    
    def save(self, path: str) -> bool:
        try:
            t.save(self.model.state_dict(), path)
            return True
        except Exception:
            return False
