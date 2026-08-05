class NLPPipelineBuilder50:
    """
    A data science tool for building NLP pipelines.

    Attributes:
        None

    Methods:
        build_pipeline: Builds an NLP pipeline.
    """

    def build_pipeline(self, data):
        # TO DO: implement pipeline building logic
        pass


def main():
    """
    Main function for the nlp-pipeline-builder-50 tool.

    Returns:
        None
    """
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.naive_bayes import MultinomialNB

    data = pd.read_csv('data.csv')
    X = data['text']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    pipeline = NLPPipelineBuilder50()
    pipeline.build_pipeline(data)


if __name__ == "__main__":
    main()