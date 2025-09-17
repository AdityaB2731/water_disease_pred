# # predict_from_doc.py
# import requests
# import pandas as pd
# import joblib

# # -------------------- Load Text --------------------
# with open('dummy_input.txt', 'r') as file:
#     document_text = file.read()

# # -------------------- Groq API Setup --------------------
# API_KEY = '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'  # Replace with your key
# API_URL = 'https://api.groq.com/v1/chat/completions'
# headers = {
#     'Authorization': f'Bearer {API_KEY}',
#     'Content-Type': 'application/json'
# }

# # Prompt to extract both water & patient info
# payload = {
#     "prompt": """
# You are an intelligent parser. Extract structured JSON from the input text including:
# - turbidity (float)
# - pH (float)
# - bacteria (0 or 1)
# - rainfall (float, in mm)
# - cases_last_week (int)
# - season (summer, monsoon, winter)
# - patient_fever (0 or 1)
# - patient_diarrhea (0 or 1)
# - patient_abdominal_pain (0 or 1)
# Return only a JSON object.
# """,
#     "document": document_text
# }

# response = requests.post(API_URL, headers=headers, json=payload)
# parsed_json = response.json()

# # -------------------- Build DataFrame --------------------
# season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
# season_dict[f"season_{parsed_json['season']}"] = 1

# input_data = pd.DataFrame([{
#     "turbidity": parsed_json['turbidity'],
#     "pH": parsed_json['pH'],
#     "bacteria_presence": parsed_json['bacteria'],
#     "rainfall": parsed_json['rainfall'],
#     "cases_last_week": parsed_json['cases_last_week'],
#     "patient_fever": parsed_json['patient_fever'],
#     "patient_diarrhea": parsed_json['patient_diarrhea'],
#     "patient_abdominal_pain": parsed_json['patient_abdominal_pain'],
#     **season_dict
# }])

# # Ensure correct feature order
# training_columns = joblib.load('training_columns.pkl')
# input_data = input_data.reindex(columns=training_columns, fill_value=0)

# # -------------------- Load Model & Predict --------------------
# model = joblib.load('disease_predictor.pkl')
# prediction = model.predict(input_data)[0]
# probability = model.predict_proba(input_data).max()

# print(f"Predicted Disease: {prediction} (Probability: {probability:.2f})")
# predict_from_doc.py
# import requests
# import pandas as pd
# import joblib
# import json

# # -------------------- Load Text --------------------
# with open('dummy_input.txt', 'r') as file:
#     document_text = file.read()

# # -------------------- Groq API Setup --------------------
# API_KEY = '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'  # Replace with your key
# API_URL = "https://api.groq.com/openai/v1/chat/completions"

# headers = {
#     'Authorization': f'Bearer {API_KEY}',
#     'Content-Type': 'application/json'
# }

# # -------------------- Prepare Groq Payload --------------------
# payload = {
#     "model": "llama-3.3-70b-versatile",
#     "messages": [
#         {"role": "system", "content": "You are an intelligent parser."},
#         {"role": "user", "content": f"Extract structured JSON from the following text including:\n"
#                                      "- turbidity (float)\n"
#                                      "- pH (float)\n"
#                                      "- bacteria (0 or 1)\n"
#                                      "- rainfall (float, in mm)\n"
#                                      "- cases_last_week (int)\n"
#                                      "- season (summer, monsoon, winter)\n"
#                                      "- patient_fever (0 or 1)\n"
#                                      "- patient_diarrhea (0 or 1)\n"
#                                      "- patient_abdominal_pain (0 or 1)\n\n"
#                                      f"Text:\n\"\"\"{document_text}\"\"\""}
#     ]
# }

# # -------------------- Send Request --------------------
# response = requests.post(API_URL, headers=headers, json=payload)
# if response.status_code != 200:
#     print(f"Status code: {response.status_code}")
#     print(f"Response text: {response.text}")
#     exit(1)

# result = response.json()

# # -------------------- Parse JSON --------------------
# try:
#     raw_json = result['choices'][0]['message']['content']
#     # Remove any markdown or code blocks from Groq response
#     raw_json = raw_json.strip('` \n')
#     parsed_json = json.loads(raw_json)
# except (KeyError, json.JSONDecodeError) as e:
#     print("Failed to parse response:", e)
#     print("Raw response:", result)
#     exit(1)

# # -------------------- Build DataFrame --------------------
# season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
# season_dict[f"season_{parsed_json['season']}"] = 1

# input_data = pd.DataFrame([{
#     "turbidity": parsed_json['turbidity'],
#     "pH": parsed_json['pH'],
#     "bacteria": parsed_json['bacteria'],
#     "rainfall": parsed_json['rainfall'],
#     "cases_last_week": parsed_json['cases_last_week'],
#     "patient_fever": parsed_json['patient_fever'],
#     "patient_diarrhea": parsed_json['patient_diarrhea'],
#     "patient_abdominal_pain": parsed_json['patient_abdominal_pain'],
#     **season_dict
# }])

# # -------------------- Ensure correct feature order --------------------
# training_columns = joblib.load('training_columns.pkl')
# input_data = input_data.reindex(columns=training_columns, fill_value=0)

# # -------------------- Load Model & Predict --------------------
# model = joblib.load('disease_predictor.pkl')
# prediction = model.predict(input_data)[0]
# probability = model.predict_proba(input_data).max()

# print(f"Predicted Disease: {prediction} (Probability: {probability:.2f})")
# import requests
# import pandas as pd
# import joblib
# import json
# import re

# # -------------------- Load Text --------------------
# with open('dummy_input.txt', 'r') as file:
#     document_text = file.read()

# # -------------------- Groq API Setup --------------------
# API_KEY = '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'
# API_URL = "https://api.groq.com/openai/v1/chat/completions"

# headers = {
#     'Authorization': f'Bearer {API_KEY}',
#     'Content-Type': 'application/json'
# }

# # -------------------- Prepare Groq Payload --------------------
# payload = {
#     "model": "llama-3.3-70b-versatile",
#     "messages": [
#         {"role": "system", "content": "You are an intelligent parser."},
#         {"role": "user", "content": f"Extract structured JSON from the following text including:\n"
#                                      "- turbidity (float)\n"
#                                      "- pH (float)\n"
#                                      "- bacteria (0 or 1)\n"
#                                      "- rainfall (float, in mm)\n"
#                                      "- cases_last_week (int)\n"
#                                      "- season (summer, monsoon, winter)\n"
#                                      "- patient_fever (0 or 1)\n"
#                                      "- patient_diarrhea (0 or 1)\n"
#                                      "- patient_abdominal_pain (0 or 1)\n\n"
#                                      f"Text:\n\"\"\"{document_text}\"\"\""}
#     ]
# }

# # -------------------- Send Request --------------------
# response = requests.post(API_URL, headers=headers, json=payload)
# if response.status_code != 200:
#     print(f"Status code: {response.status_code}")
#     print(f"Response text: {response.text}")
#     exit(1)

# result = response.json()

# # -------------------- Parse JSON from the model's message --------------------
# try:
#     # Get the content of the assistant's message
#     raw_content = result['choices'][0]['message']['content']
    
#     # Extract JSON from code block if present
#     match = re.search(r"```json\n(.*?)\n```", raw_content, re.DOTALL)
#     if match:
#         parsed_json = json.loads(match.group(1))
#     else:
#         # fallback: try loading the content directly
#         parsed_json = json.loads(raw_content.strip())
    
#     print("Extracted JSON:", parsed_json)
# except (KeyError, json.JSONDecodeError) as e:
#     print("Failed to parse JSON:", e)
#     print("Raw message content:", raw_content)
#     exit(1)

# # -------------------- Build DataFrame --------------------
# season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
# season_dict[f"season_{parsed_json['season']}"] = 1

# input_data = pd.DataFrame([{
#     "turbidity": parsed_json['turbidity'],
#     "pH": parsed_json['pH'],
#     "bacteria": parsed_json['bacteria'],
#     "rainfall": parsed_json['rainfall'],
#     "cases_last_week": parsed_json['cases_last_week'],
#     "patient_fever": parsed_json['patient_fever'],
#     "patient_diarrhea": parsed_json['patient_diarrhea'],
#     "patient_abdominal_pain": parsed_json['patient_abdominal_pain'],
#     **season_dict
# }])

# # -------------------- Ensure correct feature order --------------------
# training_columns = joblib.load('training_columns.pkl')
# input_data = input_data.reindex(columns=training_columns, fill_value=0)

# # -------------------- Load Model & Predict --------------------
# model = joblib.load('disease_predictor.pkl')
# prediction = model.predict(input_data)[0]
# probability = model.predict_proba(input_data).max()

# print(f"Predicted Disease: {prediction} (Probability: {probability:.2f})")
import streamlit as st
import requests
import pandas as pd
import joblib
import json
import re

st.title("💧 Water Quality & Disease Outbreak Predictor")

document_text = st.text_area("Paste input text (containing water & patient info):", height=300)

if st.button("Predict Disease"):
    if not document_text.strip():
        st.error("Please enter some input text!")
    else:
        # Groq API setup
        API_KEY = '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'
        API_URL = "https://api.groq.com/openai/v1/chat/completions"

        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are an intelligent parser."},
                {"role": "user", "content": f"Extract structured JSON from the following text including:\n"
                                             "- turbidity (float)\n"
                                             "- pH (float)\n"
                                             "- bacteria (0 or 1)\n"
                                             "- rainfall (float, in mm)\n"
                                             "- cases_last_week (int)\n"
                                             "- season (summer, monsoon, winter)\n"
                                             "- patient_fever (0 or 1)\n"
                                             "- patient_diarrhea (0 or 1)\n"
                                             "- patient_abdominal_pain (0 or 1)\n\n"
                                             f"Text:\n\"\"\"{document_text}\"\"\""}
            ]
        }

        response = requests.post(API_URL, headers=headers, json=payload)

        if response.status_code != 200:
            st.error(f"API error: {response.text}")
        else:
            result = response.json()

            try:
                raw_content = result['choices'][0]['message']['content']
                match = re.search(r"```json\n(.*?)\n```", raw_content, re.DOTALL)
                if match:
                    parsed_json = json.loads(match.group(1))
                else:
                    parsed_json = json.loads(raw_content.strip())

                st.subheader("✅ Extracted JSON from Input:")
                st.json(parsed_json)

                # Build DataFrame
                season_dict = {"season_summer": 0, "season_monsoon": 0, "season_winter": 0}
                season_dict[f"season_{parsed_json['season']}"] = 1

                input_data = pd.DataFrame([{
                    "turbidity": parsed_json['turbidity'],
                    "pH": parsed_json['pH'],
                    "bacteria_presence": parsed_json['bacteria'],
                    "rainfall": parsed_json['rainfall'],
                    "cases_last_week": parsed_json['cases_last_week'],
                    "patient_fever": parsed_json['patient_fever'],
                    "patient_diarrhea": parsed_json['patient_diarrhea'],
                    "patient_abdominal_pain": parsed_json['patient_abdominal_pain'],
                    **season_dict
                }])

                training_columns = joblib.load('training_columns.pkl')
                input_data = input_data.reindex(columns=training_columns, fill_value=0)

                model = joblib.load('disease_predictor.pkl')
                accuracy = joblib.load('model_accuracy.pkl')
                prediction = model.predict(input_data)[0]
                probability = model.predict_proba(input_data).max()
                label_encoder = joblib.load('label_encoder.pkl')
                metrics = joblib.load('model_metrics.pkl')
                predicted_label = label_encoder.inverse_transform([prediction])[0]
                st.info(f"Accuracy: {metrics['accuracy']*100:.2f}%")
                st.info(f"Precision: {metrics['precision']:.4f}")
                st.info(f"Recall: {metrics['recall']:.4f}")
                st.info(f"F1 Score: {metrics['f1']:.4f}")
                # st.text(metrics['report'])
                # st.info(f"📊 Model Accuracy: {metrics['accuracy'] * 100:.2f}%")
                # st.info(f"🎯 Precision: {metrics['precision']:.4f}")
                # st.info(f"🔧 Recall: {metrics['recall']:.4f}")
                # st.info(f"🔔 F1 Score: {metrics['f1_score']:.4f}")
                st.subheader("📋 Classification Report:")
                # st.text(metrics['classification_report'])
                st.success(f"🩺 Predicted Disease: {predicted_label} (Probability: {probability:.2f})")
                st.success(f"🩺 Predicted Disease: {prediction} (Probability: {probability:.2f})")

            except Exception as e:
                st.error(f"Failed to parse Groq response: {e}")
