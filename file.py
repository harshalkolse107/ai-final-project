from transformers import pipeline

# Specify the model explicitly
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

posts = [
    "The recent changes in the app have made it much easier to use!",
    "I didn’t find the update helpful at all; it made things more complicated.",
    "The support team was so responsive and helped solve my issues quickly!",
    "This product is overpriced and doesn’t meet my expectations."
]

# Analyze sentiment for each post
for post in posts:
    result = sentiment_model(post)[0]
    print(f"Text: {post}")
    print(f"Sentiment: {result['label']} with score {result['score']:.2f}\n")

