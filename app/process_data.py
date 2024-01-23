# Importing required modules
import pandas as pd
from joblib import load
import os
import shutil
from datetime import datetime

# Specify your directory path
dir_path = './data'
archive_path = './archive'
result_path = './result'

def process():
    # Load data from csv file in the specified directory
    df = pd.read_csv(os.path.join(dir_path, 'data.csv'))

    # Separate features and target
    X = df[['X1', 'X2']].values
    y = df['y'].values

    # Load the model from the file
    model = load('regression_lineaire.joblib')

    # Make predictions
    predictions = model.predict(X)

    # Create a DataFrame for the predictions
    df_predictions = pd.DataFrame(predictions, columns=['predictions'])

    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    # Move original data to archive with timestamp
    if not os.path.exists(archive_path):
        os.makedirs(archive_path)
    shutil.move(os.path.join(dir_path, 'data.csv'), os.path.join(archive_path, f'data_{timestamp}.csv'))

    # Save predictions to result with timestamp
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    df_predictions.to_csv(os.path.join(result_path, f'result_{timestamp}.csv'), index=False)
    return df_predictions