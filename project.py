from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import json
from joblib import dump

data_file = "All_Beauty.jsonl"


# main function
def json_to_dataframe(json_data):
    data_list = []
    counter = 0
    for line in json_data.split("\n"):
        counter += 1
        if line.strip() != "":
            try:
                data = json.loads(line)
                data_list.append(
                    {"rating": data["rating"], "title": data["title"], "text": data["text"]}
                )
            except json.JSONDecodeError as e:
                print(f"An error occurred: {e}")
                print(f"An error occurred: {e.__traceback__}")
    df = pd.DataFrame(data_list)
    print(f"Total records: {counter}")
    return df


if __name__ == '__main__':
    json_data = open(data_file, "r").read()
    df = json_to_dataframe(json_data)

    print(df.head())

    # Combine 'title' and 'text' columns
    df['title_text'] = df['title'] + " " + df['text']

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(df['title_text'], df['rating'], test_size=0.2, random_state=42)

    # Create a pipeline that first transforms the 'title_text' into feature vectors then trains the classifier
    text_clf = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MLPClassifier(verbose=True)),
    ])

    # Train the classifier
    text_clf.fit(X_train, y_train)
    # save the model
    try:
        dump(text_clf, 'model.joblib')
    except Exception as e:
        print(f"An error occurred in saving the model: {e}")
    # Predict the test set results
    y_pred = text_clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Model accuracy: {accuracy}")
