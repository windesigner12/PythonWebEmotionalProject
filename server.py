"""
Flask web server to detect emotions in text using IBM Watson NLP API.
"""

from flask import Flask, render_template, request
from EmotionDetection import emotion_detector

app = Flask(__name__)

@app.route('/')
def home():
    """
    Render the home page with a form to submit text.
    """
    return render_template('index.html')

@app.route('/emotionDetector', methods=['GET'])
def detect_emotion():
    """
    Process the text input from the frontend and return the emotion analysis result.
    """
    text_to_analyze = request.args.get('textToAnalyze', '').strip()

    if not text_to_analyze:
        return "Invalid text! Please try again!"

    try:
        result = emotion_detector(text_to_analyze)

        if result['dominant_emotion'] is None:
            return "Invalid text! Please try again!"

        formatted_response = (
            f"For the given statement, the system response is "
            f"'anger': {result['anger']}, "
            f"'disgust': {result['disgust']}, "
            f"'fear': {result['fear']}, "
            f"'joy': {result['joy']} and "
            f"'sadness': {result['sadness']}. "
            f"The dominant emotion is {result['dominant_emotion']}."
        )

        return formatted_response

    except Exception as error:  # pylint: disable=broad-exception-caught
        return f"Error occurred: {str(error)}", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
