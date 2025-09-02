
from flask import Flask, render_template, request, url_for

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = joblib.load('fake_review_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# Initialize NLTK tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        user_review = request.form['review']
        cleaned = clean_text(user_review)
        vectorized = tfidf.transform([cleaned])
        pred = model.predict(vectorized)[0]
        if pred == 1:       # Adjust this if your dataset uses 0/1 or -1/1
            prediction = "Fake Review"
        else:
            prediction = "Genuine Review"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
