
---

# Bank Customer Churn Prediction App

## Introduction

This application is designed to predict the likelihood of a bank customer churning (i.e., leaving the bank). The prediction is based on several customer attributes, such as credit score, age, tenure, balance, and more. The app leverages a trained Random Forest Classifier model to make accurate predictions.

## Features

- **Credit Score**: The credit score of the customer.
- **Age**: The age of the customer.
- **Tenure**: The number of years the customer has been with the bank.
- **Balance**: The account balance of the customer.
- **Number of Products**: The number of products the customer has purchased.
- **Has Credit Card**: Whether the customer has a credit card (1 for Yes, 0 for No).
- **Is Active Member**: Whether the customer is an active member (1 for Yes, 0 for No).
- **Estimated Salary**: The estimated salary of the customer.
- **Geography**: The country of the customer (France, Germany, or Spain).
- **Gender**: The gender of the customer (Female or Male).

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/Dau2004/bank-customer-churn-prediction.git
    cd bank-customer-churn-prediction
    ```

2. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

## Usage

1. **Open the app**: After running the above command, a local server will start, and you can view the app in your browser.

2. **Enter customer details**: Use the input fields on the web page to enter the details of the customer you want to predict churn for.

3. **Predict**: Click the "Predict" button to see the prediction. The app will display whether the customer is at risk of churn or not.

## Model Training and Evaluation

### Data Preprocessing

- **Dropping unnecessary columns**: Removed columns such as 'RowNumber', 'CustomerId', and 'Surname'.
- **Handling categorical variables**: Converted categorical variables like 'Geography' and 'Gender' into dummy/indicator variables.
- **Feature scaling**: Standardized features by removing the mean and scaling to unit variance.

### Model Training

Several models were trained, including Logistic Regression, K-Nearest Neighbors, Support Vector Machines, Decision Trees, and Random Forests. The Random Forest Classifier was chosen as the best model based on its performance.

### Model Tuning

GridSearchCV was used to find the best hyperparameters for the Random Forest model.

### Feature Selection

RFECV was employed to select the most important features, improving model performance and reducing complexity.

### Model Evaluation

The final model was evaluated using accuracy, precision, recall, F1-score, and AUC-ROC.

## Results

- **Accuracy**: 0.8580
- **Precision**: 0.7444
- **Recall**: 0.4224
- **F1-score**: 0.5390
- **AUC-ROC**: 0.6935

## File Structure

- `app.py`: Main file to run the Streamlit app.
- `requirements.txt`: List of required Python packages.
- `best_model1.pkl`: Trained model saved using joblib.
- `selected_features.json`: JSON file containing the list of selected features.

## Contributing

Contributions are welcome! Please fork the repository and submit pull requests for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

## Acknowledgements

Thanks to the creators of the libraries and tools used in this project, including Pandas, Scikit-learn, Streamlit, and Joblib.

---

