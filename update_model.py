import joblib
from sklearn.ensemble import RandomForestClassifier

# Load your model file
model = joblib.load('random_forest_classifier.pkl')

# Re-save the model using the current version
joblib.dump(model, 'random_forest_classifier_updated.pkl')

