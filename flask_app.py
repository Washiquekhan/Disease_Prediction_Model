from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = pickle.load(open('model/disease_model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))

# Symptoms list (shortened for demo)
symptoms = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
            'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue']

@app.route('/')
def index():
    return render_template('index.html', symptoms=symptoms)

@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')
    input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]
    prediction = model.predict([input_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    return render_template('result.html', disease=disease, symptoms=selected_symptoms)

if __name__ == "__main__":
    app.run(debug=True)