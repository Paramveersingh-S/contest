import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

class AutoencoderModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 8),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec

class Autoencoder:
    def __init__(self, input_dim: int, epochs: int = 100, lr: float = 1e-3):
        self.model = AutoencoderModel(input_dim)
        self.epochs = epochs
        self.lr = lr
        
    def fit(self, X_train: np.ndarray, X_val: np.ndarray = None):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32) if X_val is not None else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(X_train_t)
            loss = criterion(out, X_train_t)
            loss.backward()
            optimizer.step()
            
            if X_val_t is not None:
                self.model.eval()
                with torch.no_grad():
                    val_out = self.model(X_val_t)
                    val_loss = criterion(val_out, X_val_t).item()
                    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter > 10:
                    break
                    
    def score(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            out = self.model(X_t)
        # MSE per sample
        mse = torch.mean((X_t - out)**2, dim=1).numpy()
        return mse
        
    def save(self, model_dir: str):
        os.makedirs(model_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(model_dir, 'ae_weights.pt'))
        
    def load(self, model_dir: str):
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'ae_weights.pt')))
        self.model.eval()
