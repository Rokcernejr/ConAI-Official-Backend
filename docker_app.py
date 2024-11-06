# app.py
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, TFBertModel
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging

# Suppress warnings
logging.getLogger("transformers.modeling_tf_utils").setLevel(logging.ERROR)

app = Flask(__name__)

# Load BERT tokenizer and model for text processing
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Define and load model architecture
def create_model(input_shape):
    input_layer = Input(shape=(input_shape,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = BatchNormalization()(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Helper function to preprocess data
def preprocess_data(data):
    data.dropna(inplace=True)
    data['Class_Name'] = data['Material_Name'].str.split().str[0]
    min_samples_per_class = 2
    class_counts = data['Class_Name'].value_counts()
    classes_to_keep = class_counts[class_counts >= min_samples_per_class].index
    data = data[data['Class_Name'].isin(classes_to_keep)]
    return data

# Training endpoint
@app.route('/train', methods=['POST'])
def train():
    # Load and preprocess data
    file = request.files['file']
    data = pd.read_excel(file, sheet_name='Sheet1', header=None)
    data.columns = ['Material_Name', 'Lead_Time']
    data = preprocess_data(data)
    
    # Encode labels and prepare features
    X = data['Material_Name'].apply(lambda x: bert_tokenizer(x, return_tensors='tf', padding=True, truncation=True)['input_ids'][0])
    X = tf.stack(X.values)
    y = pd.factorize(data['Class_Name'])[0]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model and train
    model = create_model(X_train.shape[1:])
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32, callbacks=[early_stopping, reduce_lr])
    
    # Model evaluation
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # Save metrics and visualizations
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'classification_report': report
    }
    
    # Confusion matrix plot
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Convert plot to base64 for API response
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    return jsonify({
        'metrics': metrics,
        'confusion_matrix_plot': plot_url
    })

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    data = pd.DataFrame(input_data)
    data = preprocess_data(data)
    
    X = data['Material_Name'].apply(lambda x: bert_tokenizer(x, return_tensors='tf', padding=True, truncation=True)['input_ids'][0])
    X = tf.stack(X.values)
    
    predictions = model.predict(X)
    predicted_classes = (predictions > 0.5).astype("int32")
    
    response = {
        'predictions': predicted_classes.tolist()
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
