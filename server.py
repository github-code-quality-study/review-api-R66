import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from urllib.parse import parse_qs
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

# Download necessary NLTK data
nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load reviews from CSV file
reviews = pd.read_csv('data/reviews.csv').to_dict('records')
TIMESTAMP_FORMAT = '%Y-%m-%d %H:%M:%S'

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        pass

    def analyze_sentiment(self, review_body: str) -> dict:
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        method = environ['REQUEST_METHOD']

        if method == "GET":
            query_params = parse_qs(environ['QUERY_STRING'])
            location = query_params.get('location', [None])[0]
            start_date = query_params.get('start_date', [None])[0]
            end_date = query_params.get('end_date', [None])[0]

            filtered_reviews = reviews

            if location:
                filtered_reviews = [review for review in filtered_reviews if review['Location'] == location]

            if start_date:
                try:
                    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) >= start_date_dt
                    ]
                except ValueError:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid start_date format. Use YYYY-MM-DD"}).encode("utf-8")]

            if end_date:
                try:
                    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
                    filtered_reviews = [
                        review for review in filtered_reviews
                        if datetime.strptime(review['Timestamp'], TIMESTAMP_FORMAT) <= end_date_dt
                    ]
                except ValueError:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Invalid end_date format. Use YYYY-MM-DD"}).encode("utf-8")]

            for review in filtered_reviews:
                review['sentiment'] = self.analyze_sentiment(review['ReviewBody'])

            filtered_reviews = sorted(
                filtered_reviews, 
                key=lambda x: x['sentiment']['compound'], 
                reverse=True
            )

            response_body = json.dumps([
                {
                    "ReviewId": review["ReviewId"],
                    "ReviewBody": review["ReviewBody"],
                    "Location": review["Location"],
                    "Timestamp": review["Timestamp"],
                    "sentiment": review["sentiment"]
                } for review in filtered_reviews
            ], indent=2).encode("utf-8")

            start_response("200 OK", [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body)))
            ])

            return [response_body]

        elif method == "POST":
            try:
                # Read POST data from form-urlencoded
                content_length = int(environ.get('CONTENT_LENGTH', 0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                params = parse_qs(post_data)

                location = params.get('Location', [None])[0]
                review_body = params.get('ReviewBody', [None])[0]
                if location not in ["Albuquerque, New Mexico", "Carlsbad, California", "Chula Vista, California", 
                                    "Colorado Springs, Colorado", "Denver, Colorado", "El Cajon, California", "El Paso, Texas", 
                                    "Escondido, California", "Fresno, California", "La Mesa, California", "Las Vegas, Nevada", 
                                    "Los Angeles, California", "Oceanside, California", "Phoenix, Arizona", "Sacramento, California", 
                                    "Salt Lake City, Utah", "San Diego, California", "Tucson, Arizona" ]:
                    location = False

                # Validate input
                if not location or not review_body:
                    start_response("400 Bad Request", [("Content-Type", "application/json")])
                    return [json.dumps({"error": "Missing 'Location' or 'ReviewBody'"}).encode("utf-8")]

                # Generate UUID and Timestamp
                review_id = str(uuid.uuid4())
                timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)

                # Analyze sentiment
                sentiment = self.analyze_sentiment(review_body)

                # Create new review record
                new_review = {
                    "ReviewId": review_id,
                    "Location": location,
                    "Timestamp": timestamp,
                    "ReviewBody": review_body,
                }

                # Append to in-memory list
                reviews.append(new_review)

                # Prepare response
                response_body = json.dumps(new_review, indent=2).encode("utf-8")

                start_response("201 Created", [
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(response_body)))
                ])

                return [response_body]

            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "application/json")])
                return [json.dumps({"error": str(e)}).encode("utf-8")]

        else:
            start_response("405 Method Not Allowed", [("Content-Type", "text/plain")])
            return [b"Method Not Allowed"]

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = int(os.environ.get('PORT', 8000))
    with make_server("0.0.0.0", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()
