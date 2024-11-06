# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging

# Suppress warnings
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

app = Flask(__name__)

# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Create the model
def create_model(input_shape):
    input_layer = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

@app.route('/train', methods=['POST'])
def train():
    # Load and preprocess data from request
    data = pd.DataFrame(request.json['data'])
    data.columns = ['Material_Name', 'Lead_Time']
    
    # Preprocessing
    data['Class_Name'] = data['Material_Name'].str.split().str[0]
    scaler = MinMaxScaler()
    data['Lead_Time'] = scaler.fit_transform(data[['Lead_Time']])

    # Define features and labels
    X = data['Lead_Time'].values.reshape(-1, 1)
    y = np.where(data['Class_Name'] == 'SomeClass', 1, 0)  # Dummy binary target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = create_model(input_shape=X_train.shape[1])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    # Evaluation and metrics
    y_pred = model.predict(X_test).round().astype(int)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Generate visualizations
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    # Save plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return jsonify({
        "classification_report": report,
        "confusion_matrix_image": image_base64
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess input data
    input_data = pd.DataFrame(request.json['data'])
    X = input_data['Lead_Time'].values.reshape(-1, 1)

    # Make predictions
    y_pred = model.predict(X).round().astype(int)

    return jsonify({"predictions": y_pred.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
