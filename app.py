# Runtime download of required NLTK data
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Initialize the Flask app first
app = Flask(__name__)
CORS(app)

# Initialize tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def english_to_isl(text):
    # Step 1: Tokenization
    tokens = word_tokenize(text)

    # Step 2: Remove punctuation and stop words
    filtered_tokens = [
        word.lower() for word in tokens
        if word.lower() not in stop_words
           and word not in string.punctuation
    ]

    # Step 3: Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Step 4: Grammar adjustment for ISL (SOV - Subject, Object, Verb order)
    tagged_tokens = nltk.pos_tag(lemmatized_tokens)
    subject, object_, verb, others = [], [], [], []

    for word, tag in tagged_tokens:
        if tag.startswith('NN'):  # Noun: Subject or Object
            if not subject:
                subject.append(word)
            else:
                object_.append(word)
        elif tag.startswith('VB'):  # Verb
            verb.append(word)
        else:
            others.append(word)  # Any other words can be added to a separate list

    # Step 5: Reconstruct the sentence in SOV order
    isl_sentence = subject + object_ + verb + others
    return " ".join(isl_sentence)

@app.route('/convert_to_isl', methods=['POST'])
def convert_to_isl():
    data = request.json
    text = data.get('text', '')
    print(f"Received text: {text}")  # Log the incoming request

    try:
        if not text:
            raise ValueError("No text provided")
        isl_output = english_to_isl(text)
        return jsonify({
            'original_text': text,
            'isl_text': isl_output
        })
    except Exception as e:
        print(f"Error: {str(e)}")  # Log the error
        return jsonify({
            'error': str(e)
        }), 500
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load model once
interpreter = tf.lite.Interpreter(model_path="sign_language_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Characters your model predicts
CHARACTERS = [chr(i) for i in range(65, 91)]  # A-Z

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        image_b64 = data.get("image_base64")
        if not image_b64:
            return jsonify({"error": "No image data provided"}), 400

        # Decode image
        img_bytes = base64.b64decode(image_b64)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Failed to decode image"}), 400

        # Resize & normalize
        input_img = cv2.resize(img, (64, 64))  # adjust to model input
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Run inference
        interpreter.set_tensor(input_details[0]["index"], input_img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]["index"])[0]

        # Get top prediction
        top_idx = np.argmax(output)
        confidence = output[top_idx]
        predicted_char = CHARACTERS[top_idx] if confidence >= 0.5 else "?"

        return jsonify({"output": predicted_char, "confidence": float(confidence)})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/")
def index():
    return "TFLite Sign Language Inference API is live."

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

