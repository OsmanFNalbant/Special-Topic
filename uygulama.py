#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tkinter as tk
from tkinter import messagebox
import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import numpy as np
from keras.models import load_model
from joblib import load

# Modeli yükle
model_path = 'C:\\Users\\osman\\Desktop\\Notebook Workspace\\emotion_model.h5'  # Model dosyanızın yolu
model = load_model(model_path)

# Scaler dosyasını yükle
scaler_file = 'C:\\Users\\osman\\Desktop\\Notebook Workspace\\scaler.pkl'  # Scaler dosyasının yolu
loaded_scaler = load(scaler_file)

# Ses kaydetme fonksiyonu
def record_audio(duration=2.5, sample_rate=22050, file_name='recorded_audio.wav'):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Kayıt bitene kadar bekle
    write(file_name, sample_rate, (audio_data * 32767).astype(np.int16))  # WAV dosyası olarak kaydet
    print(f"Recording finished. Saved as {file_name}")
    return file_name

# Özellik çıkarım fonksiyonu
def extract_features_for_test(file_path, loaded_scaler, sample_rate=22050):
    data, sr = librosa.load(file_path, duration=2.5, offset=0.6)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=20)
    mfcc_processed = np.mean(mfcc.T, axis=0)
    mfcc_processed = mfcc_processed.reshape(1, -1)
    mfcc_scaled = loaded_scaler.transform(mfcc_processed)
    mfcc_scaled = np.expand_dims(mfcc_scaled, axis=2)
    return mfcc_scaled

# Tahmin fonksiyonu
def test_audio_model(file_path, model, loaded_scaler):
    features = extract_features_for_test(file_path, loaded_scaler)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    predicted_emotion = emotion_labels[predicted_class[0]]
    return predicted_emotion, prediction[0]

# UI fonksiyonu
def run_prediction():
    file_name = record_audio(duration=2.5, sample_rate=22050, file_name='recorded_audio.wav')
    predicted_emotion, confidence_scores = test_audio_model(file_name, model, loaded_scaler)

    result_text = f"Predicted Emotion: {predicted_emotion}\n"
    result_text += "Confidence Scores:\n"
    emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    for label, score in zip(emotion_labels, confidence_scores):
        result_text += f"{label}: {score:.4f}\n"

    messagebox.showinfo("Prediction Result", result_text)

# Tkinter GUI
root = tk.Tk()
root.title("Emotion Recognition")

# Record Button
record_button = tk.Button(root, text="Record and Predict", command=run_prediction, height=2, width=20)
record_button.pack(pady=20)

# Start the GUI loop
root.mainloop()

