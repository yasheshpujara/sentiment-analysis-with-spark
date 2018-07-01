from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover

from nltk.corpus import stopwords

from preprocessing import remove_punctuation
from word_normalization import tf_idf, word_to_vector
from models import *


def clean_data():
    """
    Clean the Tweet by removing punctuations and stop words
    :return cleaned data:
    """
    data = sc.textFile("data/data.txt")
    col_rdd = data.map(lambda x: (x.split('\t')[0], x[-1]))
    punctuation_removed_rdd = col_rdd.map(lambda x: (remove_punctuation(x[0]), float(x[1])))

    data_df = sqlContext.createDataFrame(punctuation_removed_rdd, ["text", "label"])
    remover = StopWordsRemover(inputCol="text", outputCol="words", stopWords=stopwords.words('english'))
    return remover.transform(data_df).select(["label", "words"])


def show_stats(predicted_df):
    """
    Display accuracy and confusion matrix for the models
    :param predicted_df: predicted dataframe of test data
    """
    accuracy = calculate_accuracy(predicted_df)
    confusion_table = confusion_matrix(predicted_df)
    print("\nAccuracy of the model: {}".format(round(accuracy*100, 2)))
    print("\nConfusion Matrix:" + confusion_table)


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    
    filtered_data_df = clean_data()
    training, test = filtered_data_df.rdd.randomSplit([0.7, 0.3], seed=0)
    train_df = tf_idf(training)
    test_df = tf_idf(test)

    nb_predicted_df = naive_bayes_classifier(train_df, test_df)
    print("\nNaive Bayes classifier\n")
    show_stats(nb_predicted_df)

    lor_predicted_df = logistic_regression_classifier(train_df, test_df)
    print("\nLogistic Regression classifier\n")
    show_stats(lor_predicted_df)

    # TODO: Apply models on twitter data for sentiment analysis