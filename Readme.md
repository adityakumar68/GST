# GST Hackthon

This project implements an ensemble voting classifier to predict a binary target variable. The repository consists of two main Jupyter notebooks that handle different aspects of the machine learning pipeline.

## Project Structure

1. **`GST_model_building.ipynb`**: This notebook contains the complete model building process, including:
   - Data preprocessing
   - Model selection
   - Hyperparameter tuning
   - Creation of the final ensemble model

2. **`Validation_data_prediction.ipynb`**: This notebook is designed for making predictions on new, unseen data. It:
   - Loads the trained model and preprocessing objects
   - Applies these to new data to generate predictions

## Making Predictions on New Data

To predict the target variable for your validation data, follow these steps:

1. **Prepare your validation data:** 
   - Ensure your data is in CSV format
   - The data should have the same columns (features) as the training data, excluding the target variable column

2. **Open `Validation_data_prediction.ipynb`**

3. **Update the data path:**
   - In the second cell of the notebook, locate the following line:
     ```python
     X_test = pd.read_csv('Test_20/X_Test_Data_Input.csv')
     ```
   - Replace `'Test_20/X_Test_Data_Input.csv'` with the path to your validation data CSV file

4. **Run the second cell:** 
   - This cell loads the pre-trained model, preprocesses your validation data, makes predictions, and saves the results

5. **Check the output files:** 
   Two output files will be created in the same directory as the notebook:
   - `predictions_output.csv`: Contains the full validation dataset with two additional columns:
     - `Predicted_Class`: The predicted binary class
     - `Predicted_Probability`: The probability of belonging to the positive class
   - `predicted_classes_only.csv`: Contains only the predicted classes (0,1) for your validation data

## Important Notes

### Required Files

Ensure that the following files, are in the same directory as `Validation_data_prediction.ipynb`:

- `voting_classifier_model.joblib`
- `cat_imputer.joblib`
- `num_imputer.joblib`
- `scaler.joblib`

### Dependencies

Make sure you have the necessary Python libraries installed. You can install them using the following command:

```bash
pip install scikit-learn catboost xgboost lightgbm joblib pandas
```

## Additional Information

For complete details on the model development process, refer to `GST_model_building.ipynb`. This notebook provides a comprehensive overview of the steps taken to arrive at the optimal prediction model.