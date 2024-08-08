import pandas as pd
import numpy as np
import argparse
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _parse_args():
    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    parser.add_argument('--filepath', type=str, default='/opt/ml/processing/input/')
    parser.add_argument('--filename', type=str, default='iris.csv')
    parser.add_argument('--outputpath_train', type=str, default='/opt/ml/processing/output/train/')
    parser.add_argument('--outputpath_test', type=str, default='/opt/ml/processing/output/test/')
    
    return parser.parse_known_args()

if __name__=="__main__":
    args, _ = _parse_args()
    logger.info("Arguments parsed. Filepath: %s, Filename: %s, Outputpath Train: %s, Outputpath Test: %s", args.filepath, args.filename, args.outputpath_train, args.outputpath_test)

    df = pd.read_csv(os.path.join(args.filepath, args.filename))
    logger.info("Data loaded from %s", os.path.join(args.filepath, args.filename))
    
    # Preprocessing
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    df.dropna(inplace=True)  # Drop rows with missing values
    logger.info("Preprocessing completed. Data columns: %s", df.columns)
    
    y = df["target"]
    
    # Train-Test Split
    train_data, test_data = train_test_split(df, test_size=0.2, stratify=y)
    logger.info("Train-test split completed. Train size: %d, Test size: %d", len(train_data), len(test_data))

    # Save to local CSV files
    os.makedirs(args.outputpath_train, exist_ok=True)
    os.makedirs(args.outputpath_test, exist_ok=True)

    train_file = os.path.join(args.outputpath_train, 'iris_train.csv')
    test_file = os.path.join(args.outputpath_test, 'iris_test.csv')
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    logger.info("Train and test data saved to %s and %s", train_file, test_file)

    print("## Preprocessing and uploading completed. Exiting.")
