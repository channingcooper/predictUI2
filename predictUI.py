import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_rmspe(y_true, y_pred):
    return 100 * np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def calculate_mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

def preprocess_input(input_data, encoders, non_numeric_cols, imputer, numeric_cols):
    input_df = pd.DataFrame([input_data])
    for col in non_numeric_cols:
        if col in input_df.columns:
            input_df[col] = encoders[col].transform(input_df[col].astype(str))
    if not set(numeric_cols).issubset(input_df.columns):
        st.write(f"Numeric columns mismatch: {set(numeric_cols) - set(input_df.columns)}")
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    return input_df

def main():
    st.title("Machine Learning Model UI")

    # Upload CSV file
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        # Check if 'Settlement_Amount' is in the DataFrame
        if 'Settlement_Amount' not in data.columns:
            st.error("The column 'Settlement_Amount' is not in the uploaded dataset.")
            return

        # Identify non-numeric columns and convert them to numeric types or encode them
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns

        # Encode non-numeric columns and store the encoders
        encoders = {}
        for col in non_numeric_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

        # Handle missing values
        numeric_cols = data.select_dtypes(include=[np.number]).columns.drop('Settlement_Amount')
        imputer = SimpleImputer(strategy='median')
        imputed_data = imputer.fit_transform(data[numeric_cols])
        data[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols)

        # Generate and plot the correlation matrix
        corr_matrix = data.corr()
        plt.figure(figsize=(14, 10))
        sns.set(style='white')
        ax = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 14}, cbar=False)
        plt.title('Correlation Matrix', fontsize=16, color='white')
        plt.xticks(rotation=45, ha='right', fontsize=14, color='white')
        plt.yticks(rotation=0, fontsize=14, color='white')
        plt.gcf().set_facecolor('black')
        plt.gca().set_facecolor('black')
        plt.tight_layout()
        st.pyplot(plt)

        # Split the data into features and target variable
        X = data.drop(columns=['Settlement_Amount'])
        y = data['Settlement_Amount']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Define the models and their hyperparameters for GridSearchCV
        model_params = {
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            }
        }

        # Perform GridSearchCV for each model
        best_models = {}
        for name, mp in model_params.items():
            clf = GridSearchCV(mp['model'], mp['params'], cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
            clf.fit(X_train, y_train)
            best_models[name] = clf.best_estimator_
            st.write(f"Best parameters for {name}: {clf.best_params_}")

        # Evaluate the best models
        results = {}
        for name, model in best_models.items():
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            rmspe_value = calculate_rmspe(y_test, y_pred)
            mape_value = calculate_mape(y_test, y_pred)
            results[name] = {
                'Mean Squared Error': mse,
                'Mean Absolute Error': mae,
                'R-squared': r2,
                'RMSPE': rmspe_value,
                'MAPE': mape_value
            }
            st.write(f"{name} - MSE: {mse}, MAE: {mae}, R²: {r2}, RMSPE: {rmspe_value}, MAPE: {mape_value}")

            # Plot the predicted vs actual values
            st.subheader(f'{name} Predictions')
            comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
            st.write(comparison_df.head())
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, label='Predicted', color='gold', alpha=1)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
            ax.set_xlabel('Actual Settlement Amount', color='white', fontsize=14)
            ax.set_ylabel('Predicted Settlement Amount', color='white', fontsize=14)
            ax.set_title(f'{name}: Actual vs Predicted Settlement Amount', color='white', fontsize=16)
            ax.legend()
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.set_facecolor('black')
            fig.set_facecolor('black')
            st.pyplot(fig)

            # Calculate SHAP values
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            # SHAP summary plot
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
            st.pyplot(fig)

        # Single prediction form
        st.sidebar.header("Single Prediction")
        single_input = {}
        for feature in X.columns:
            value = st.sidebar.text_input(f"Enter value for {feature}", "")
            if value:  # Only process if the input is not empty
                if feature in non_numeric_cols:
                    single_input[feature] = value
                else:
                    try:
                        single_input[feature] = float(value)
                    except ValueError:
                        st.sidebar.error(f"Invalid input for {feature}. Please enter a numeric value.")

        if st.sidebar.button("Predict"):
            if len(single_input) < len(X.columns):
                st.sidebar.error("Please enter values for all features.")
            else:
                input_df = preprocess_input(single_input, encoders, non_numeric_cols, imputer, numeric_cols)
                input_features = input_df

                predictions = {}
                for name, model in best_models.items():
                    predictions[name] = model.predict(input_features)[0]

                # Ensemble prediction (average of all models)
                ensemble_prediction = np.mean(list(predictions.values()))
                predictions['Ensemble'] = ensemble_prediction

                st.sidebar.subheader("Prediction Results")
                for name, pred in predictions.items():
                    st.sidebar.write(f"{name}: {pred}")

if __name__ == '__main__':
    main()
