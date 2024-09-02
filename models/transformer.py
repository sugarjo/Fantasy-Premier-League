# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:21:46 2024

@author: jorgels
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np


# Identify float columns
float_cols = objective_X.select_dtypes(include=['float64']).columns

# Replace NaNs with the mean of the respective columns
objective_X[float_cols] = objective_X[float_cols].apply(lambda col: col.fillna(col.median()))

X = objective_X
y = train_y

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['category']).columns
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns

# Preprocessing: one-hot encode categorical variables and standardize numerical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_preprocessed = preprocessor.fit_transform(X)

# Reshape the data to (num_observations, sequence_length, num_features)
# Assume each observation is treated as a sequence of length 1
X_dense = X_preprocessed.toarray()
X_expanded = np.expand_dims(X_dense, axis=1)  # Now shape is (num_observations, 1, num_features)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_expanded, y, test_size=0.2, random_state=42)
X_fit, X_eval, y_fit, y_eval = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LayerNormalization, MultiHeadAttention, Dropout, Flatten

# Define the Transformer block
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Define the Transformer model
def create_transformer_model(input_shape, embed_dim, num_heads, ff_dim):
    inputs = Input(shape=input_shape)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(inputs)
    x = Flatten()(x)  # Flatten to (batch_size, features)
    outputs = Dense(1)(x)  # Single float output
    return tf.keras.Model(inputs=inputs, outputs=outputs)

input_shape = (1, X_fit.shape[2])  # sequence_length=1, num_features from preprocessed data

embed_dim = X_fit.shape[2]  # Embedding dimension (same as num_features)
num_heads = 8  # Number of attention heads
ff_dim = 512  # Feed-forward layer dimension

model = create_transformer_model(input_shape, embed_dim, num_heads, ff_dim)
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
history = model.fit(X_fit, y_fit, epochs=40, validation_data=(X_eval, y_eval))

# Evaluate the model
loss = model.evaluate(X_val, y_val)
print(f"Validation Loss: {loss}")

# Make predictions
predictions = model.predict(X_val)
print(predictions[:5])  # Print first 5 predictions



