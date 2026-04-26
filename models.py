import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


class CNN_Softmax:
    def __init__(self, input_dim, num_classes):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, X, y, epochs=5):
        print("Training CNN...")

        try:
            X = torch.tensor(X).float().to(self.device)
            y = torch.tensor(y).long().to(self.device)
        except RuntimeError as exc:
            if self.device.type != "cuda":
                raise
            print(f"CUDA failed during tensor transfer, falling back to CPU: {exc}")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            X = torch.tensor(X).float().to(self.device)
            y = torch.tensor(y).long().to(self.device)

        for epoch in range(epochs):
            try:
                outputs = self.model(X)
            except RuntimeError as exc:
                if self.device.type != "cuda":
                    raise
                print(f"CUDA failed during training, falling back to CPU: {exc}")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                X = X.to(self.device)
                y = y.to(self.device)
                outputs = self.model(X)

            loss = self.criterion(outputs, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    def fit(self, X, y):
        self.train(X, y)

    def predict(self, X):
        self.model.eval()

        try:
            X = torch.tensor(X).float().to(self.device)
        except RuntimeError as exc:
            if self.device.type != "cuda":
                raise
            print(f"CUDA failed during inference setup, falling back to CPU: {exc}")
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            X = torch.tensor(X).float().to(self.device)

        with torch.no_grad():
            try:
                outputs = self.model(X)
            except RuntimeError as exc:
                if self.device.type != "cuda":
                    raise
                print(f"CUDA failed during inference, falling back to CPU: {exc}")
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
                X = X.to(self.device)
                outputs = self.model(X)

            _, preds = torch.max(outputs, 1)

        return preds.cpu().numpy()


class ELM_Model:
    def __init__(self, input_dim, hidden_dim=256):
        self.W = np.random.randn(input_dim, hidden_dim)
        self.b = np.random.randn(hidden_dim)

    def _activation(self, X):
        return np.maximum(0, X)

    def train(self, X, y):
        H = self._activation(X @ self.W + self.b)
        Y = np.eye(len(np.unique(y)))[y]
        self.beta = np.linalg.pinv(H) @ Y

    def fit(self, X, y):
        self.train(X, y)

    def predict(self, X):
        H = self._activation(X @ self.W + self.b)
        return np.argmax(H @ self.beta, axis=1)


class MLP_Model:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(256,))

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


def get_model(mode, input_dim=None, num_classes=None, params=None):
    if mode in ["baseline", "temporal"]:
        return CNN_Softmax(input_dim, num_classes)

    if mode == "hybrid_svm":
        C = params.get("C", 1) if params else 1
        return SVC(kernel="linear", C=C)

    if mode == "hybrid_elm":
        return ELM_Model(input_dim)

    if mode == "skeleton":
        return MLP_Model()

    if mode == "multimodal":
        C = params.get("C", 1) if params else 1
        return SVC(kernel="linear", C=C)

    raise ValueError(f"Unknown mode: {mode}")
