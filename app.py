from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# Initialize the sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    score = None
    if request.method == "POST":
        text = request.form["text"]
        # Analyze sentiment of input text
        result = sentiment_model(text)[0]
        sentiment = result['label']
        score = result['score']

    return render_template("index.html", sentiment=sentiment, score=score)

if __name__ == "__main__":
    app.run(debug=True)
