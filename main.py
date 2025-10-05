import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

df = pd.read_csv(r"C:\Users\user\Documents\med_nn\hospital data analysis.csv")

# Extracting features and targets
conditions = df['conditions'].values
procedures = df['procedures'].values

print(f"Loaded {len(conditions)} samples from dataset")
print(f"Unique procedures: {len(np.unique(procedures))}")

# Embedding Model (BERT-based) 
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Converting condition text to dense semantic vectors
print("Encoding conditions to embeddings...")
X = np.array([embedder.encode(text) for text in tqdm(conditions.tolist(),desc="Embedding Text",ncols=100)])

# Encoding procedures into numbers
encoder = LabelEncoder()
y = encoder.fit_transform(procedures)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# neural network layers
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

    def backward(self, grad_output):
        return grad_output

class CrossEntropyLoss:
    def compute_loss(self, predictions, true_labels):
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        one_hot = np.zeros_like(predictions)
        one_hot[np.arange(len(true_labels)), true_labels] = 1
        losses = -np.sum(one_hot * np.log(predictions), axis=1)
        return np.mean(losses)

# neural network model
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

# model initialization
input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y))

model = SimpleFeedForward(input_size, hidden_size, output_size)

# training configuration
epochs = 1000
learning_rate = 0.1
momentum = 0.9
l2_lambda = 0.01

print("\n" + "="*50)
print("Starting Training...")
print("="*50)

# training model
for epoch in tqdm (range(epochs),desc="Training Progress",ncols=100):
    predictions = model.forward(X_train)
    loss = model.compute_loss(predictions, y_train)
    model.optimize(predictions, y_train, learning_rate, momentum, l2_lambda)

    if epoch % 100 == 0:
        # Calculating training accuracy
        train_pred_labels = np.argmax(predictions, axis=1)
        train_acc = np.mean(train_pred_labels == y_train)
        tqdm.write(f"Epoch {epoch:4d} - Loss: {loss:.4f} - Train Acc: {train_acc*100:.2f}%")

# final evaluation
print("\n" + "="*50)
print("Evaluation Results")
print("="*50)

test_predictions = model.forward(X_test)
predicted_labels = np.argmax(test_predictions, axis=1)
accuracy = np.mean(predicted_labels == y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Correct predictions: {np.sum(predicted_labels == y_test)}/{len(y_test)}")

# interactive chatbot
print("\n" + "="*50)
print("Medical Procedure Prediction System")
print("="*50)
print("Enter a medical condition to get a suggested procedure")
print("Type 'quit' to exit\n")

while True:
    user_input = input("Enter a medical condition: ")
    if user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Encoding user input
    input_vector = embedder.encode([user_input])
    prediction = model.forward(np.array(input_vector))
    predicted_index = np.argmax(prediction)
    predicted_procedure = encoder.inverse_transform([predicted_index])
    
    # Getting confidence score
    confidence = prediction[0][predicted_index] * 100

    print(f"Suggested procedure: {predicted_procedure[0]}")
    print(f"Confidence: {confidence:.2f}%\n")