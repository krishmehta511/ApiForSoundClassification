from flask import Flask
import tensorflow as tf
# import librosa
import numpy as np
import pickle

app = Flask(__name__)


labelencoder = pickle.load(open('static/label_encoder.pkl', 'rb'))
model = tf.keras.models.load_model('static/SavedModel')


# def getFeatures(audioInput):
#     audioData, audioSampleRate = librosa.load(audioInput, res_type="kaiser_fast")
#     features = librosa.feature.mfcc(y=audioData, sr=audioSampleRate, n_mfcc=40)
#     scaled_features = np.mean(features.T, axis=0)
#     return scaled_features


@app.route('/')
def index():
    return 'hello'



if __name__ == "__main__":
    app.run()