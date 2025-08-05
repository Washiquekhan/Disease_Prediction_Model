import pickle
import numpy as np

# Load model and encoder
model = pickle.load(open('model/disease_model.pkl', 'rb'))
le = pickle.load(open('model/label_encoder.pkl', 'rb'))

# List of symptoms (short list for demo)
symptoms = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering',
            'chills','joint_pain','stomach_pain','acidity','ulcers_on_tongue']

def predict_disease(input_symptoms):
    input_vector = [1 if symptom in input_symptoms else 0 for symptom in symptoms]
    prediction = model.predict([input_vector])[0]
    return le.inverse_transform([prediction])[0]

if __name__ == "__main__":
    symptoms_input = ['itching', 'skin_rash']
    result = predict_disease(symptoms_input)
    print(f"Predicted Disease: {result}")