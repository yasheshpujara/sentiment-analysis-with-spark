# ABOUT THE DATA

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015.

It contains sentences labelled with positive or negative sentiment, extracted from reviews of products, movies, and restaurants

## FORMAT

* sentence \t score \n

## DETAILS

* Score is either 1 (for positive) or 0 (for negative)	
* The sentences come from three different websites/fields:

    * imdb.com
    * amazon.com
    * yelp.com

* For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews.
* File "data.txt" contains in total 3000 labelled sentences, 1500 for each category.
* Only sentences are selected which have a clearly positive or negative connotation, the goal was for no neutral sentences to be selected.

## FURTHER INFORMATION

For the full datasets look:

* imdb: Maas et. al., 2011 'Learning word vectors for sentiment analysis'
* amazon: McAuley et. al., 2013 'Hidden factors and hidden topics: Understanding rating dimensions with review text'
* yelp: Yelp dataset challenge http://www.yelp.com/dataset_challenge
