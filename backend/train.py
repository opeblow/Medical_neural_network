import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "hospital data analysis.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.weight_velocity = np.zeros_like(self.weights)
        self.bias_velocity = np.zeros_like(self.biases)

    def forward(self, input_data):
        self.input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def optimize(self, grad_output, learning_rate, momentum=0.9, l2_lambda=0.01):
        weight_gradient = np.dot(self.input.T, grad_output) + l2_lambda * self.weights
        bias_gradient = np.sum(grad_output, axis=0, keepdims=True)
        self.weight_velocity = momentum * self.weight_velocity - learning_rate * weight_gradient
        self.bias_velocity = momentum * self.bias_velocity - learning_rate * bias_gradient
        self.weights += self.weight_velocity
        self.biases += self.bias_velocity

class ReLU:
    def forward(self, input_data):
        self.input = input_data
        return np.maximum(0, input_data)

    def backward(self, grad_output):
        grad_input = grad_output.copy()
        grad_input[self.input <= 0] = 0
        return grad_input

class Softmax:
    def forward(self, input_data):
        exp_vals = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

class CrossEntropyLoss:
    def compute_loss(self, predictions, true_labels):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(len(true_labels)), true_labels] = 1
        losses = -np.sum(one_hot * np.log(predictions), axis=1)
        return np.mean(losses)

class SimpleFeedForward:
    def __init__(self, n_inputs, hidden_size, output_size):
        self.dense1 = DenseLayer(n_inputs, hidden_size)
        self.relu = ReLU()
        self.dense2 = DenseLayer(hidden_size, output_size)
        self.softmax = Softmax()
        self.loss_fn = CrossEntropyLoss()

    def forward(self, X):
        self.out1 = self.dense1.forward(X)
        self.out2 = self.relu.forward(self.out1)
        self.out3 = self.dense2.forward(self.out2)
        self.out4 = self.softmax.forward(self.out3)
        return self.out4

    def compute_loss(self, predictions, labels):
        return self.loss_fn.compute_loss(predictions, labels)

    def optimize(self, predictions, labels, learning_rate, momentum=0.9, l2_lambda=0.01):
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(len(labels)), labels] = 1
        grad_output2 = (predictions - one_hot) / predictions.shape[0]
        self.dense2.optimize(grad_output2, learning_rate, momentum, l2_lambda)
        grad_hidden = np.dot(grad_output2, self.dense2.weights.T)
        grad_hidden = self.relu.backward(grad_hidden)
        self.dense1.optimize(grad_hidden, learning_rate, momentum, l2_lambda)


def train():
    print("=" * 50)
    print("Medical Procedure Prediction - Training")
    print("=" * 50)

    df = pd.read_csv(DATA_PATH)
    conditions = df["Condition"].values
    procedures = df["Procedure"].values

    print(f"Loaded {len(conditions)} samples")
    print(f"Unique procedures: {len(np.unique(procedures))}")

    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Encoding conditions to embeddings...")
    X = np.array([embedder.encode(text) for text in tqdm(conditions.tolist(), desc="Embedding Text", ncols=100)])

    encoder = LabelEncoder()
    y = encoder.fit_transform(procedures)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(np.unique(y))

    model = SimpleFeedForward(input_size, hidden_size, output_size)

    epochs = 1000
    learning_rate = 0.1
    momentum = 0.9
    l2_lambda = 0.01

    print("\nStarting Training...")
    for epoch in tqdm(range(epochs), desc="Training Progress", ncols=100):
        predictions = model.forward(X_train)
        loss = model.compute_loss(predictions, y_train)
        model.optimize(predictions, y_train, learning_rate, momentum, l2_lambda)

        if epoch % 100 == 0:
            train_pred_labels = np.argmax(predictions, axis=1)
            train_acc = np.mean(train_pred_labels == y_train)
            print(f"Epoch {epoch:4d} - Loss: {loss:.4f} - Train Acc: {train_acc*100:.2f}%")

    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    test_predictions = model.forward(X_test)
    predicted_labels = np.argmax(test_predictions, axis=1)
    accuracy = np.mean(predicted_labels == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    model_data = {
        "dense1_weights": model.dense1.weights,
        "dense1_biases": model.dense1.biases,
        "dense2_weights": model.dense2.weights,
        "dense2_biases": model.dense2.biases,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
    }
    model_path = os.path.join(MODEL_DIR, "model.pkl")
    encoder_path = os.path.join(MODEL_DIR, "encoder.pkl")
    embedder_path = os.path.join(MODEL_DIR, "embedder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)
    with open(embedder_path, "wb") as f:
        pickle.dump(embedder, f)

    print(f"\nModel saved to: {model_path}")
    print(f"Encoder saved to: {encoder_path}")
    print(f"Embedder saved to: {embedder_path}")
    print("\nTraining complete!")


if __name__ == "__main__":
    train()
