# Stockify - Smart Inventory Management System

A machine learning-powered inventory management system for retail stores in Nagpur.

## Features

- Synthetic dataset generation with realistic features
- Deep learning model for demand prediction
- Interactive Streamlit dashboard
- Inventory analysis and alerts
- Store and product-specific insights

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate the dataset:
```bash
python generate_dataset.py
```

3. Train the model:
```bash
python model.py
```

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## Components

- `generate_dataset.py`: Creates synthetic retail data
- `model.py`: Deep learning model for demand prediction
- `app.py`: Streamlit web interface
- `requirements.txt`: Project dependencies
