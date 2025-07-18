#   P y t h o n W e b E m o t i o n a l P r o j e c t 
 # 🎭 Emotion Detection Web Application

 This project is a Flask-based web application that detects the emotional tone behind written text using IBM Watson NLP's Emotion Prediction API.

 ---

 ## 🚀 Features

- 🔍 Detects five key emotions: **anger**, **disgust**, **fear**, **joy**, and **sadness**
- 🧠 Uses Watson NLP's pre-trained emotion model
- 🧪 Includes unit tests for reliable emotion classification
- 🌐 User-friendly web interface for quick testing
- 📦 Packaged as a Python module for reusability

---

## 📁 Project Structure
 
final_project/
│
├── EmotionDetection/ # Python package
│ ├── init.py # Package initializer
│ └── emotion_detection.py # Main detection logic
│
├── static/
│ └── mywebscript.js # Frontend JavaScript (provided)
│
├── templates/
│ └── index.html # Frontend HTML interface (provided)
│
├── test_emotion_detection.py # Unit tests
├── server.py # Flask web server
├── setup.py # Package setup configuration
├── LICENSE
└── README.md # Project documentation
