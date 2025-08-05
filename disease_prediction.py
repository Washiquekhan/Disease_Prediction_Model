import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('dataset/Training.csv')

# Split data
X = df.drop(columns=['prognosis'])
y = df['prognosis']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and label encoder
pickle.dump(model, open('model/disease_model.pkl', 'wb'))
pickle.dump(le, open('model/label_encoder.pkl', 'wb'))