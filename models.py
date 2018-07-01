import numpy as np
import pandas as pd

from pyspark.ml.classification import NaiveBayes, LogisticRegression


def naive_bayes_classifier(training_df, testing_df):
    """
    Apply Naive Bayes Classifier to test data for predicting sentiment of Tweets.
    :param training_df: Trained labelled data
    :param testing_df: Test data
    :return: transformed dataframe of predicted labels for tweets
    """
    nb = NaiveBayes()
    model = nb.fit(training_df)
    return model.transform(testing_df).select(["label", "words", "prediction"])


def logistic_regression_classifier(training_df, testing_df):
    """
    Apply Logistic Regression Classifier to test data for predicting sentiment of Tweets.
    :param training_df: Trained labelled data
    :param testing_df: Test data
    :return: transformed dataframe of predicted labels for tweets
    """
    lor = LogisticRegression(regParam=0.01)
    model = lor.fit(training_df)
    return model.transform(testing_df).select(["label", "words", "prediction"])


def calculate_accuracy(result_df):
    """
    Calculate accuracy of model against actual data.
    :param result_df: Dataframe returned from the model
    :return: accuracy between 0 and 1
    """
    return 1.0 * result_df.filter(result_df.label == result_df.prediction).count() / result_df.count()


def confusion_matrix(result_df):
    """
    Generate Confusion Matrix for showing the performance of algorithm.
    :param result_df: Dataframe returned from the model
    :return: pandas dataframe
    """
    true_positives = result_df.filter((result_df.label == 1.0) & (result_df.prediction == 1.0)).count()
    true_negatives = result_df.filter((result_df.label == 0.0) & (result_df.prediction == 0.0)).count()
    false_positives = result_df.filter((result_df.label == 0.0) & (result_df.prediction == 1.0)).count()
    false_negatives = result_df.filter((result_df.label == 1.0) & (result_df.prediction == 0.0)).count()

    matrix = {"Positive": pd.Series([true_positives, false_positives], index=["Positive", "Negative"]),
              "Negative": pd.Series([false_negatives, true_negatives], index=["Positive", "Negative"])}

    df = pd.DataFrame(matrix)
    df.columns.name = "Actual / Predicted"
    return df
