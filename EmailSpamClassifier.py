import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess the dataset
def load_and_preprocess_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Print the column names to identify the correct names
    print("Column Names:", data.columns)

    # Update the column names based on your dataset
    # Replace 'message' and 'label' with the actual names 'Message' and 'Category'
    train_data, test_data, train_labels, test_labels = train_test_split(
        data['Message'], data['Category'], test_size=0.2, random_state=42
    )

    # Convert text data to numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)

    return train_features, test_features, train_labels, test_labels


    # Convert text data to numerical features using CountVectorizer
    vectorizer = CountVectorizer()
    train_features = vectorizer.fit_transform(train_data)
    test_features = vectorizer.transform(test_data)

    return train_features, test_features, train_labels, test_labels

# Function to train and evaluate the classifier
def train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, algorithm):
    # Train the classifier
    classifier = algorithm()
    classifier.fit(train_features, train_labels)

    # Make predictions on the test set
    predictions = classifier.predict(test_features)

    # Evaluate the performance
    accuracy = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions)

    return accuracy, report

# Function to compare different algorithms
def compare_algorithms(file_path):
    # Load and preprocess the dataset
    train_features, test_features, train_labels, test_labels = load_and_preprocess_data(file_path)

    # Define algorithms to compare
    algorithms = [MultinomialNB, DecisionTreeClassifier]

    # Compare algorithms
    for algorithm in algorithms:
        print(f"\nResults for {algorithm.__name__}:")
        accuracy, report = train_and_evaluate_classifier(train_features, test_features, train_labels, test_labels, algorithm)
        print(f"Accuracy: {accuracy}")
        print("Classification Report:\n", report)

# Specify the path to your dataset CSV file
# You can find datasets on Kaggle, UCI Machine Learning Repository, or use your custom dataset
dataset_path = 'D:\PROGRAMMING\Email Spam Classifier\\Datasets.csv'

# Call the function to compare algorithms
compare_algorithms(dataset_path)
