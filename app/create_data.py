# Importing required modules
import numpy as np
import os
import pandas as pd

def create():
    # Create data
    X = np.random.randn(10, 2)
    y = 0.1 + 0.1 * X[:,0] - 0.2 * X[:, 1]

    # Convert to DataFrame for easier csv export
    df = pd.DataFrame(np.column_stack((X,y)), columns=['X1', 'X2', 'y'])

    # Specify your directory path
    dir_path = './data'

    # Create directory if it doesn't exist
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Save data to csv file in the specified directory
    df.to_csv(os.path.join(dir_path, 'data.csv'), index=False)
    return df