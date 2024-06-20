import joblib 
 
try: 
    model = joblib.load('random_forest_classifier.pkl') 
    print("Model loaded successfully") 
except FileNotFoundError: 
    print("The file was not found.") 
except ModuleNotFoundError as e: 
    print(f"Module not found: {e}") 
except Exception as e: 
    print(f"An error occurred: {e}") 
