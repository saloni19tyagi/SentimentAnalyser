import csv
import random
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import FreqDist
from bs4 import BeautifulSoup
from nltk.classify.naivebayes import NaiveBayesClassifier
import nltk


def get_reviews():

    print("Reading Reviews from Disk: Started")

    # Read Movie Reviews Corpus
    file_handler = open("data/IMDBDataset.csv", "r")
    reader = csv.reader(file_handler)

    # Read Reviews into Memory & Shuffle them
    next(reader)
    reviews = [(BeautifulSoup(document, ).get_text(), category) for document, category in reader]
    # reviews = reviews[0:100]
    random.shuffle(reviews)

    print("Reading Reviews from Disk: Complete")

    return reviews


def get_top_words(reviews, count):

    print("Generate top features to use: Started")

    # Convert Document into words
    words = [word.lower()
             for (document, category) in reviews
             for word in document.split()]

    # Remove Stop Words
    tmp = []
    stop_words = set(stopwords.words('english'))
    for word in words:
        if word not in stop_words:
            tmp.append(word)
    words = tmp

    # Do Stemming
    porter = PorterStemmer()
    tmp = [porter.stem(word) for word in words]
    words = tmp

    # Generate Frequency Distribution and get Top X words
    most_common_words = FreqDist(w for w in words).most_common(count)

    print(f"Generate top features to use: Completed, Feature Count: {count}")

    return most_common_words


def get_features(document, all_feature_list):
    features = {}
    for word in document.split():
        if word in all_feature_list:
            if word not in features:
                features[word] = 1
            # else:
            #     features[word] += 1

    return features


def start():

    reviews = get_reviews()
    top_words = [i[0] for i in get_top_words(reviews, 2000)]

    # Generate Features Sets
    print ("Generate Feature_set for all documents: Started")
    feature_set = []
    for review, category in reviews:
        feature_set.append((get_features(review, top_words), category))

    print("Generate Feature_set for all documents: Completed")

    test_set, train_set = feature_set[:20000], feature_set[20000:]

    print("Training Started")
    classifier = NaiveBayesClassifier.train(train_set)
    print("Training Started")

    print("Testing Now....")
    print(nltk.classify.accuracy(classifier, test_set))
    # print(classifier.show_most_informative_features())


start()
