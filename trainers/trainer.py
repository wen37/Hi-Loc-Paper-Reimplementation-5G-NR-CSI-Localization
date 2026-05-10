import torch
import os
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.checkpoint import save_checkpoint
from utils.visualization import plot_loss_curves, save_json

class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model.to(config['train']['device'])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config['train']['lr'],
            weight_decay=config["train"].get("weight_decay", 0.0),
        )
        self.device = config['train']['device']
        self.grad_clip = config["train"].get("grad_clip", None)
        self.best_val_loss = float("inf")
        self.early_stop_patience = config["train"].get("early_stop_patience", 10)
        self.checkpoint_path = config["train"].get(
            "checkpoint_path", "checkpoints/best_model.pt"
        )
        self.output_dir = config["train"].get("output_dir", "outputs")
        self.history = {"train_loss": [], "val_loss": []}
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for x, y in tqdm(self.train_loader, desc="Training"):
            x, y = x.to(self.device), y.to(self.device)
            
            self.optimizer.zero_grad()
            pred = self.model(x)
            loss = self.criterion(pred, y)
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            
            total_loss += loss.item()
        return total_loss / len(self.train_loader)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                total_loss += loss.item()
        return total_loss / len(self.val_loader)
    
    def fit(self):
        epochs = self.config['train']['epochs']
        patience_counter = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch + 1,
                    val_loss,
                    self.checkpoint_path,
                )
                print(
                    f"Saved best checkpoint to {self.checkpoint_path} "
                    f"(val_loss={val_loss:.4f})"
                )
            else:
                patience_counter += 1

            if patience_counter >= self.early_stop_patience:
                print(
                    f"Early stopping triggered after {epoch+1} epochs. "
                    f"Best val loss: {self.best_val_loss:.4f}"
                )
                break

        history_dir = os.path.join(self.output_dir, "train")
        save_json(self.history, os.path.join(history_dir, "loss_history.json"))
        plot_loss_curves(
            self.history["train_loss"],
            self.history["val_loss"],
            os.path.join(history_dir, "loss_curve.png"),
        )
