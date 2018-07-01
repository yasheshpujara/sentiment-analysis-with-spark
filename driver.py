from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext


if __name__ == "__main__":
    conf = SparkConf()
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    # TODO: Load labelled data

    # TODO: Pre-processing of the data

    # TODO: Train classification models

    # TODO: Apply models on twitter data for sentiment analysis