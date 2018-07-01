import string


def remove_punctuation(sentence):
    """
    Remove punctuations from the sentence
    :param sentence:
    :return: list of words without punctuation
    """
    punctuations = list(string.punctuation)
    extra_punctuations = ['.', '``', '...', '\'s', '--', '-', 'n\'t', '_', '–']
    punctuations += extra_punctuations
    filtered = [w for w in sentence.lower() if w not in punctuations]
    return ("".join(filtered)).split()
