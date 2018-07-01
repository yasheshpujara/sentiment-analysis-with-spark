from pyspark.ml.feature import HashingTF, IDF, Word2Vec


def tf_idf(data_rdd):
    """
    Calculate term frequency–inverse document frequency for reflecting importance of words in Tweet.
    :param data_rdd: input data rdd
    :return: transformed dataframe
    """
    data_rdd_df = data_rdd.toDF()
    hashing_tf = HashingTF(inputCol="words", outputCol="tf_features")
    tf_data = hashing_tf.transform(data_rdd_df)

    idf_data = IDF(inputCol="tf_features", outputCol="features").fit(tf_data)
    tf_idf_data = idf_data.transform(tf_data)
    return tf_idf_data.select(["label", "words", "features"])


def word_to_vector(data_rdd):
    """
    Vectorization of words in a sentence
    :param data_rdd: input data rdd
    :return: transformed dataframe
    """
    data_rdd_df = data_rdd.toDF()
    word2vec = Word2Vec(inputCol="words", outputCol="features")
    model = word2vec.fit(data_rdd_df)
    word2vec_data = model.transform(data_rdd_df)
    return word2vec_data.select(["label", "words", "features"])
