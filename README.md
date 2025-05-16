# 🛒 Stockify – Store Demand Forecasting

**Stockify** is a machine learning-based demand forecasting tool designed to help retail store owners predict future product demand. It utilizes advanced models like SARIMA, Prophet, and Deep Learning (Keras) to generate accurate forecasts based on historical sales data.

---

## 🚀 Features

- 📈 Demand forecasting using:
  - SARIMA
  - Facebook Prophet
  - Deep Learning (Keras)
- 🧠 Pre-trained models for instant predictions
- 📊 CSV-based data input
- 🔄 Handles seasonality, trends, and categorical encodings
- 💾 SQLite database support
- 🖥️ Flask backend ready for deployment

---

## 📁 Project Structure

├── app.py # Flask web application
├── data_upload.py # Handles CSV upload
├── database.py # SQLite operations
├── model.py # Model loading and inference
├── generate_dataset.py # Generates synthetic dataset
├── requirements.txt # Dependencies list
├── retail_data.csv # Retail dataset
├── retail_data_historical.csv # Historical dataset
├── stockify.db # SQLite DB file
├── demand_model_dl.keras # Deep Learning model
├── demand_model_sarima.joblib # SARIMA model
├── demand_model_prophet.json # Prophet model
├── demand_model_prophet_history.csv
├── demand_model_encoders.joblib
├── demand_model_scalers.joblib
├── README.md # Documentation

yaml
Copy
Edit

---

## 🛠️ Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/PriyanshBastawale11/Stockify---Store-Demand-Forcasting.git
cd Stockify---Store-Demand-Forcasting
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv .venv
.\.venv\Scripts\activate  # Windows
Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Run the application:

bash
Copy
Edit
python app.py
📊 Use Cases
Retail inventory optimization

Multi-store sales trend analysis

Over/understock prediction

Seasonal planning and festival demand estimation

🔮 Future Improvements
Streamlit/Gradio-based UI

Real-time sales data integration

Enhanced model accuracy with external factors

User authentication and dashboard

