#  Medical Procedure Prediction System

This project is a *Neural Network–based medical chatbot* that predicts the most likely medical procedure for a given condition.  
It uses a *BERT-based SentenceTransformer* to convert medical condition text into dense semantic embeddings,  
and a *custom-built feedforward neural network (from scratch using NumPy)* to predict the appropriate procedure.



## Features

- Uses *SentenceTransformer (all-MiniLM-L6-v2)* for semantic embeddings  
- Custom *Feedforward Neural Network* with:
  - ReLU activation  
  - Softmax output  
  - Cross-Entropy loss  
  - L2 Regularization  
  - Momentum-based optimization  
- Trains directly on your CSV dataset (conditions, procedures columns)  
- Includes an *interactive chatbot interface* for real-time medical condition input and procedure suggestions  
- Displays *confidence scores* for predictions  



##  Project Structure



##  Dataset Format

Your CSV file should have *two columns*:

| conditions      | procedures       |
|-----------------|------------------|
| headache        | acetaminophen    |
| fever           | antibiotics      |
| cough           | cough syrup      |
| chest pain      | ECG              |

Save it as your_dataset.csv inside a data/ folder or in the same directory as main.py.



##  Model Overview

1. Sentence Embeddings: 
   Converts medical condition text into 384-dimensional vectors using a pre-trained BERT model.

2. Feedforward Neural Network: 
   - Input Layer → Dense (384 × 64)  
   - Hidden Layer → ReLU Activation  
   - Output Layer → Dense (64 × N_classes) + Softmax  
   - Optimized using *momentum* and *L2 regularization*

3. Training Loop: 
   Runs for a configurable number of epochs with progress logging every 100 iterations.

4. Evaluation:
   Computes test accuracy and correct prediction counts after training.

5. Chatbot Interaction:  
   Takes user input (a medical condition), embeds it, predicts the procedure, and shows confidence.



##  Installation

### 1️ Clone the repository
```bash
git clone https://github.com/opeblow/Medical_neural_network.git
cd medical_chatbot_nn

 pip install -r requirements.t numpy
pandas
scikit-learn
sentence-transformers
tqdm
Starting Training...

==================================================
Epoch 0    - Loss: 1.3863 - Train Acc: 25.00%
Epoch 100  - Loss: 0.7421 - Train Acc: 88.00%
...

Evaluation Results
==================================================
Test Accuracy: 92.50%
Correct predictions: 37/40

Medical Procedure Prediction System
==================================================
Enter a medical condition to get a suggested procedure
Type 'quit' to exit

Enter a medical condition: fever
Suggested procedure: antibiotics
Confidence: 97.25%