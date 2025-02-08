# -*- coding: utf-8 -*-
"""Lstm

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NqOQMwN4hSk3tDLoYt2hLHaFkxE7h2CH

# **BI-DIRECTIONAL LSTM OUTPUT**
"""

import numpy as np
import pandas as pd

df =pd.read_csv('sample_data/fraud_call.csv')

df.columns=["Label","Text"]
df=df.drop_duplicates()
df=df.dropna()

print(df.head())

df['Label'] = df['Label'].map({'fraud': 1, 'normal': 0})

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

texts = df["Text"].astype(str).tolist()
labels = df["Label"].tolist()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

maxlen=max([len(seq) for seq in sequences])

X=pad_sequences(sequences,maxlen=maxlen,padding='post')

y = np.array(labels)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout,GRU,Layer,Flatten

print(df['Label'].value_counts())

from tensorflow.keras.layers import Bidirectional

from tensorflow.keras import layers,models

model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1,output_dim=128, input_shape=(maxlen,)),  # Fixed input shape
    Bidirectional(LSTM(128, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(64)),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])


model.build(input_shape=(None, maxlen))


model.summary()

import tensorflow as tf

precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', precision, recall, F1Score()]
)

from sklearn.model_selection import train_test_split

xt, X_temp, yt, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

history = model.fit(
    xt, yt,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10
)



results1=model.evaluate(X_val,y_val)

print(f" validation Loss: {results1[0]}")
print(f"validation  Accuracy: {results1[1]}")
print(f" validation Precision: {results1[2]}")
print(f"validation  Recall: {results1[3]}")

results2=model.evaluate(X_test,y_test)

print(f"Test Loss: {results2[0]}")
print(f"Test Accuracy: {results2[1]}")
print(f"Test Precision: {results2[2]}")
print(f"Test Recall: {results2[3]}")

results3=model.evaluate(xt,yt)

print(f"Train Loss: {results3[0]}")
print(f"Train Accuracy: {results3[1]}")
print(f"Train Precision: {results3[2]}")
print(f"Train Recall: {results3[3]}")

def classfiy(text):
  sequence=tokenizer.texts_to_sequences([text])
  padded_sequence=pad_sequences(sequence,maxlen=maxlen,padding='post')
  prediction=model.predict(padded_sequence,verbose=0)[0][0]
  print(prediction)
  if prediction>0.5:
    return 'fraud'
  else:
    return 'normal'

model.save("model.h5")

print(classfiy('Todays Vodafone numbers ending with 4882 are selected to a receive a £350 award. If your number matches call 09064019014 to receive your £350 award'))

print(classfiy("Oh that was a forwarded message. I thought you send that to me."))

!pip install flask flask-ngrok flask-cors tensorflow numpy

from tensorflow.keras.models import load_model

modeln = load_model("model.h5")

from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
from flask_cors import CORS

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
run_with_ngrok(app)

@app.route("/")
def home():
    return "Keras ML Model API is running!"

@app.route("/predict", methods=["POST"])
def classfiy():
  data = request.json ;
  text = data["text"]
  sequence=tokenizer.texts_to_sequences([text])
  padded_sequence=pad_sequences(sequence,maxlen=maxlen,padding='post')
  prediction=modeln.predict(padded_sequence,verbose=0)[0][0]
  print(prediction)
  if prediction>0.5:
    return 'fraud'
  else:
    return 'normal'

!nohup python3 -m flask run --host=0.0.0.0 --port=5000 &

!pip install pyngrok

from pyngrok import ngrok

!ngrok authtoken 2sheSevVOgcQrpU7EmxnVevN0Oi_87V5yaUY4zaFAqCTLiaLJ
public_url = ngrok.connect(5000)
print(f"Your API is live at: {public_url}")
app.run()
