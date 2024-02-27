import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from sklearn.model_selection import KFold
import numpy as np
import datetime
import argparse

parser = argparse.ArgumentParser(description="Create web-based binary classifier for detecting disrespect.")
parser.add_argument("--dataset", type=str, default="messages-binary-classifier.csv", help="Source CSV file for the labelled messages")
parser.add_argument("--logdir", type=str, default="logs/fit", help="Where the training logs are")
args = parser.parse_args()


# track status of training using Tensorboard
log_dir = args.logdir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# update the histogram every second
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


# Load the dataset
df = pd.read_csv(args.dataset)
X = df['message']
y = df['is_respectful']  # Corrected column name

# K-Fold Cross Validation setup
num_folds = 5
kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Text vectorization parameters
max_features = 10000
sequence_length = 250

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(X):
    print(f'Training for fold {fold_no} ...')

    # Create a text vectorization layer and adapt it on the current fold's training data
    # TextVectorization under the hood strips punctuation and lowercases the text
    vectorize_layer = TextVectorization(max_tokens=max_features, output_mode='int', output_sequence_length=sequence_length)
    vectorize_layer.adapt(X.iloc[train])

    # Define the model architecture
    model = Sequential([
        vectorize_layer,
        Embedding(max_features + 1, 16),
        GlobalAveragePooling1D(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X.iloc[train], np.array(y)[train], epochs=10, verbose=1)

    # Evaluate the model on the test data
    scores = model.evaluate(X.iloc[test], np.array(y)[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

    fold_no += 1
