import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


# A1: Compute centroids and spreads for each class
def compute_centroids_and_spreads(dataframe, label_column='target'):
    class_labels = dataframe[label_column].unique()[:2]
    class_centroids = {}
    class_spreads = {}

    for class_value in class_labels:
        # Select data for the current class (excluding label column)
        class_subset = dataframe[dataframe[label_column] == class_value].drop(label_column, axis=1)
        class_centroids[class_value] = np.mean(class_subset.values, axis=0)
        class_spreads[class_value] = np.std(class_subset.values, axis=0)

    # Euclidean distance between class centroids
    inter_class_dist = np.linalg.norm(class_centroids[class_labels[0]] - class_centroids[class_labels[1]])
    return class_centroids, class_spreads, inter_class_dist


# A2: Plot histogram for a feature and return mean, variance
def plot_feature_histogram(dataframe, feature_column):
    feature_data = dataframe[feature_column]
    plt.hist(feature_data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Histogram of {feature_column}")
    plt.xlabel(feature_column)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
    return np.mean(feature_data), np.var(feature_data)


# A3: Plot Minkowski distance for r=1 to 10 between two vectors
def plot_minkowski_distance(vector_a, vector_b):
    minkowski_distances = []
    r_range = range(1, 11)
    for r in r_range:
        dist = distance.minkowski(vector_a, vector_b, p=r)
        minkowski_distances.append(dist)

    plt.plot(r_range, minkowski_distances, marker='o')
    plt.title("Minkowski Distance (r = 1 to 10)")
    plt.xlabel("r value")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()


# A5-A7: Train and evaluate kNN classifier
def train_knn_classifier(features_train, labels_train, k_neighbors=3):
    knn_model = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn_model.fit(features_train, labels_train)
    return knn_model

def test_accuracy(knn_model, features_test, labels_test):
    return knn_model.score(features_test, labels_test)

def predict_labels(knn_model, features_test):
    return knn_model.predict(features_test)


# A8: Plot accuracy vs k for kNN
def plot_accuracy_vs_k(features_train, features_test, labels_train, labels_test):
    k_range = range(1, 12)
    accuracy_scores = []
    for k in k_range:
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(features_train, labels_train)
        accuracy_scores.append(knn_model.score(features_test, labels_test))

    plt.plot(k_range, accuracy_scores, marker='o')
    plt.title("k vs Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show(block=False)
    plt.pause(3)
    plt.close()


# A9: Print confusion matrix and classification report
def evaluate_performance(knn_model, features, labels, dataset_type="Test"):
    predictions = knn_model.predict(features)
    conf_matrix = confusion_matrix(labels, predictions)
    class_report = classification_report(labels, predictions)
    print(f"\n[{dataset_type} Set] Confusion Matrix:\n", conf_matrix)
    print(f"\n[{dataset_type} Set] Classification Report:\n", class_report)


if __name__ == "__main__":
    # Load dataset
    heart_df = pd.read_csv("heart.csv")
    # Filter for binary classification (target 0 or 1)
    heart_df = heart_df[heart_df['target'].isin([0, 1])]
    # Drop 'id' column if present
    if 'id' in heart_df.columns:
        heart_df.drop(columns=['id'], inplace=True)
    print("Original age values:")
    print(heart_df['age'].head())
    # Convert age from days to years if needed
    if heart_df['age'].max() > 150:
        heart_df['age'] = heart_df['age'] // 365
    # Keep a copy of unscaled data for analysis
    unscaled_df = heart_df.copy(deep=True)

    print("Check age column (unscaled):")
    print(unscaled_df['age'].head())
    # Prepare features and labels
    features = heart_df.drop(columns=['target'])
    labels = heart_df['target']
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    # Split into train and test sets
    features_train, features_test, labels_train, labels_test = train_test_split(
        features_scaled, labels, test_size=0.3, random_state=42, stratify=labels
    )
    # Compute centroids and spreads
    centroids, spreads, inter_class_dist = compute_centroids_and_spreads(unscaled_df)
    print("Centroids computed for each class.")
    print("Spreads (Standard Deviation) computed.")
    print(f"Inter-class Euclidean distance: {inter_class_dist:.2f}")
    # Plot histogram for age
    mean_age, var_age = plot_feature_histogram(unscaled_df, 'age')
    print(f"Mean of age: {mean_age:.2f}, Variance of age: {var_age:.2f}")
    # Plot Minkowski distance between two samples
    sample_vec1 = features_scaled[0]
    sample_vec2 = features_scaled[1]
    plot_minkowski_distance(sample_vec1, sample_vec2)
    # Train kNN classifier
    knn_model = train_knn_classifier(features_train, labels_train)
    accuracy = test_accuracy(knn_model, features_test, labels_test)
    print(f"kNN Accuracy (k=3): {accuracy:.2f}")
    predictions = predict_labels(knn_model, features_test)
    print("Predictions made on test set.")
    # Plot accuracy vs k
    plot_accuracy_vs_k(features_train, features_test, labels_train, labels_test)
    # Evaluate performance on test and train sets
    evaluate_performance(knn_model, features_test, labels_test, dataset_type="Test")
    evaluate_performance(knn_model, features_train, labels_train, dataset_type="Train")