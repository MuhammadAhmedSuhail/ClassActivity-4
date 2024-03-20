from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('decision_tree_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    
    input_df = pd.DataFrame(input_data)
    
    predictions = model.predict(input_df)
    
    response = {'predictions': predictions.tolist()}
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)