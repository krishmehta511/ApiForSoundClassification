from flask import Flask
import tensorflow as tf
import pickle

app = Flask(__name__)


labelencoder = pickle.load(open('static/label_encoder.pkl', 'rb'))
model = tf.keras.models.load_model('static/SavedModel')

@app.route('/')
def index():
    return 'hello'

if __name__ == "__main__":
    app.run(debug=True)