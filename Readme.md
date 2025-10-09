# Disease Prediction System

A web-based application that predicts possible diseases based on user-reported symptoms. Built with **Flask** for the backend and **Scikit-learn** for machine learning.

---

## Features

- Symptom-based disease prediction  
- Machine learning model for accurate results  
- User-friendly web interface  
- Real-time analysis of symptoms  

---

## Project Structure

Disease-Prediction/
├── app.py # Main Flask application

├── Disease_Prediction.ipynb # Jupyter notebook for training

├── templates/ # HTML templates

│ ├── index.html

│ └── result.html

├── static/ # CSS, JS, images

├── training.csv # Training dataset

└── testing.csv # Testing dataset
---

## Getting Started

Follow these steps to set up and run the Book Recommendation System locally:

### Step 1: Clone the Repository
```bash
```bash
git clone https://github.com/supender03/Disease-Prediction.git
cd Disease-Prediction
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```
### Step 3: Run the Application
```bash
python app.py
```
### Step 4: Open in Browser
Open your web browser and go to:
```bash
http://127.0.0.1:5000/
```

How It Works

User enters symptoms via the web interface

Input is processed and formatted for the ML model

Model predicts possible diseases

Predicted results are displayed on the frontend

The ML model is trained using training.csv and evaluated with testing.csv in Disease_Prediction.ipynb.

### License

This project is licensed under the MIT License.

Feel free to use and modify it for your own projects.
