from flask import Flask, render_template, request
import numpy as np
import statistics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC 
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
app = Flask(__name__)
DATA_PATH =  "D:/project/t.csv"
data = pd.read_csv(DATA_PATH).dropna(axis = 1) 

X = data.iloc[:,:-1]
encoder = LabelEncoder() 
data["prognosis"] = encoder.fit_transform(data["prognosis"]) 
X = data.iloc[:,:-1] 
y = data.iloc[:, -1] 
X_train, X_test, y_train, y_test =train_test_split( 
X, y, test_size = 0.2, random_state = 24) 
nb_model = GaussianNB()
svm_model = SVC()
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
nb_model.fit(X_train, y_train)
svm_model.fit(X_train, y_train)





symptoms = X.columns.values

symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index

data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}

def predictDisease(symptoms):
    symptoms = symptoms.split(",")
    
    
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"].get(symptom, -1)
        if index != -1:
            input_data[index] = 1
        

   
    feature_names = list(data_dict["symptom_index"].keys())  # List of symptoms used as feature names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    input_data = np.array(input_df).reshape(1,-1)
    rf_prediction = data_dict["predictions_classes"][rf_model.predict(input_data)[0]]
    nb_prediction = data_dict["predictions_classes"][nb_model.predict(input_data)[0]]
    svm_prediction = data_dict["predictions_classes"][svm_model.predict(input_data)[0]]
    
    final_prediction = statistics.mode([rf_prediction, nb_prediction, svm_prediction])
    
    predictions = {
        "rf_model_prediction": rf_prediction,
        "naive_bayes_prediction": nb_prediction,
        "svm_model_prediction": svm_prediction,
        "final_prediction": final_prediction
    }
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.form.get('symptoms')  # Get 'symptoms' from the form
    if symptoms:
        predictions = predictDisease(symptoms)
        return render_template('index.html', predictions=predictions)
    else:
        error_message = "Please enter symptoms."
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    app.run(debug=True)
