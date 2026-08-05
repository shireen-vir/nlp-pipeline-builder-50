import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def main():
    """
    Main function for nlp-pipeline-builder-50 data science tool.
    
    This script provides basic functionality for building a natural language processing pipeline.
    It includes data loading, preprocessing, and splitting into training and testing sets.
    
    Parameters:
    None
    
    Returns:
    None
    """
    # Load the dataset
    data = pd.read_csv('data.csv')
    
    # Split the data into input text and target variable
    text = data['text']
    target = data['target']
    
    # Create a TF-IDF vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit the vectorizer to the text data and transform it into a matrix
    X = vectorizer.fit_transform(text)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42)
    
    # Print the shapes of the training and testing sets
    print("Training set shape:", X_train.shape, y_train.shape)
    print("Testing set shape:", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()