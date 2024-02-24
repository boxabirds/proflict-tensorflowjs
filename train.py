# takes the output of the generated data in main.py and turns it into a 
# tensorflow binary classifier. 

import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
import datetime
import argparse

parser = argparse.ArgumentParser(description="Create web-based binary classifier for detecting disrespect.")
parser.add_argument("--dest", type=str, default="messages-deduped.csv", help="Destination CSV file for the messages")
parser.add_argument("--logdir", type=str, default="logs/fit", help="Where the training logs are")
args = parser.parse_args()

# track status of training using Tensorboard
log_dir = args.logdir + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# update the histogram every second
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Load the dataset
df = pd.read_csv(args.dest)

# Combine 'nondisrespectful' and 'disrespectful' into a single column
df['text'] = df['nondisrespectful'].fillna('') + ' ' + df['disrespectful'].fillna('')

# Assuming 'class' column is your label: 1 for disrespectful, 0 for nondisrespectful
X = df['text'].str.lower()
y = df['class']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Text vectorization parameters
max_features = 10000  # Size of the vocabulary
sequence_length = 250  # Maximum length of each input sequence

# Create a text vectorization layer
vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Fit the text vectorization layer to the training text
vectorize_layer.adapt(X_train)

# Build the model
model = Sequential([
    vectorize_layer,
    Embedding(max_features + 1, 16),
    GlobalAveragePooling1D(),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, 
          epochs=10, 
          validation_data=(X_test, y_test),
          callbacks=[tensorboard_callback]))

model.save('respect')


