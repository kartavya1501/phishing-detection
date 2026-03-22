import joblib
import numpy as np

def predict_website(features):

    model = joblib.load("models/phishing_model.pkl")

    features = np.array(features).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    
    if prediction == 1:
        return "⚠️ Phishing Website"
    else:
        return "✅ Legitimate Website"


if __name__ == "__main__":
    sample = [80, 0, 1, 1, 2] 
    print(predict_website(sample))