import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    print("Column Names:", data.columns)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data['Message'], data['Category'], test_size=0.2, random_state=42
    )

    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)

    return train_features, test_features, train_labels, test_labels


# Training and evaluating the classifier
def train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, algorithm):
    # Training the classifier
    classifier = algorithm()
    classifier.fit(train_features, train_labels)

    predictions = classifier.predict(test_features)

    # Evaluating its performance
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    return accuracy, report

# Comparing different algorithms
def compare_algorithms(file_path):
    # Load and preprocess the dataset
    train_features, test_features, train_labels, test_labels = load_and_preprocess_data(file_path)

    algorithms = [MultinomialNB, DecisionTreeClassifier]

    # Compare algorithms
    for algorithm in algorithms:
        print(f"\nResults for {algorithm.__name__}:")
        accuracy, report = train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, algorithm)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)

dataset_path = 'D:\PROGRAMMING\Eziline Internship\Email Spam Classifier\\Datasets.csv'

compare_algorithms(dataset_path)
