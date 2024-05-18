import logging
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning

#Suppress only inconsistentversionwarnings
warnings.filterwarnings("ignore", category = InconsistentVersionWarning)

app = Flask(__name__)
ps = PorterStemmer()

# Configure logging
logging.basicConfig(level=logging.INFO)  # Set logging level to INFO

# Load NLTK stopwords
stopwords = set(stopwords.words('english'))
logging.info("NLTK stopwords loaded successfully.")

# Load model and vectorizer
try:
    model = pickle.load(open('model2.pkl', 'rb'))
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model: %s", e)

try:
    tfidfvect = pickle.load(open('tfidfvect2.pkl', 'rb'))
    logging.info("Vectorizer loaded successfully.")
except Exception as e:
    logging.error("Error loading vectorizer: %s", e)

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            text = request.form['text']
            prediction = predict(text)
            return render_template('index.html', text=text, result=prediction)
        except Exception as e:
            return render_template('index.html', error=str(e))
    return render_template('index.html')

def predict(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords]
    review = ' '.join(review)
    review_vect = tfidfvect.transform([review]).toarray()
    prediction = 'FAKE' if model.predict(review_vect) == 0 else 'REAL'
    return prediction

if __name__ == "__main__":
    app.run(debug=False)
