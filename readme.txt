Steps to Run the Project:

Save all files: Create the folder structure and save the content into the respective files.

Install dependencies:

Bash

cd smart_cab_demand
pip install -r requirements.txt
Train the model:

Bash

python demand_model.py
This will generate models/xgb_model.pkl. You might see warnings about dummy data or feature engineering, as the provided preprocess.py is a generic template.

Run the Flask API:

Bash

python app.py
The API will start on http://127.0.0.1:5000/.

Access the UI or API:

Open your browser to http://127.0.0.1:5000/ to use the simple UI.

Or use a tool like Postman/curl to send a POST request to http://127.0.0.1:5000/predict_demand with a JSON body:

JSON

{
    "timestamp": "2025-07-16 19:00:00",
    "latitude": 26.51,
    "longitude": 77.01
}
