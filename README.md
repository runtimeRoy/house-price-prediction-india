# 🏠 Indian House Price Prediction

A machine learning project that predicts house prices using the **House Price India Kaggle dataset**, with an interactive **Streamlit web application**.

## 🚀 Features

- Indian housing dataset from Kaggle
- Data cleaning and preprocessing
- Linear Regression
- Ridge Regression
- RMSE and R² evaluation
- Actual vs. predicted visualization
- Interactive Streamlit prediction interface
- GitHub-ready project structure

## 📊 Dataset

The project uses `House Price India.csv`.

The dataset contains 14,620 records and 23 columns, including:

- Number of bedrooms and bathrooms
- Living area and lot area
- Number of floors
- House condition and grade
- House and basement area
- Built and renovation year
- Postal code
- Latitude and longitude
- Nearby schools
- Distance from airport
- Price (target)

The `id` and `Date` columns are excluded from the basic model because they are identifiers/date representations rather than direct property characteristics.

## 🧠 Machine Learning Workflow

1. Load the dataset
2. Remove identifier/date fields
3. Select numerical features
4. Handle missing/infinite values
5. Split data into training and testing sets
6. Standardize features
7. Train Linear Regression and Ridge Regression models
8. Evaluate model performance
9. Use the trained models for interactive predictions

## 🌐 Run the Streamlit App

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

The app lets a user enter important house characteristics and receive estimated prices from both models.

> **Note:** For simplicity, advanced fields that are not exposed in the UI are filled with their training-set median values. This makes the app easy to use while retaining the full feature set used by the model.

## 📁 Project Structure

```text
HousePricePrediction/
│
├── House Price India.csv
├── house_price_prediction.py
├── app.py
├── house_price_prediction.png
├── requirements.txt
├── README.md
├── PROJECT_INFO.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## 🤖 AI Assistance

Claude AI was used as an AI-assisted development tool during this project. The code was reviewed and prepared for publication by the project author. The project author remains responsible for understanding, testing and presenting the final implementation.

## ⚠️ Disclaimer

This project is for educational and portfolio purposes. Predictions are model estimates and should not be treated as professional property valuations.

## 🔮 Future Improvements

- Add location-aware prediction
- Add categorical feature preprocessing
- Compare more advanced models such as Random Forest and Gradient Boosting
- Add feature-importance visualizations
- Improve model validation
- Deploy the Streamlit app online
