import argparse
import json
import os
import re
from collections import Counter

import nltk

nltk.download("stopwords")
from itertools import chain

from nltk.corpus import stopwords
from pattern.en import pluralize, singularize



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_dir",
        default="./data/COP.filt3.sub/",
        type=str,
        help="Input directory to read json files from (default ./data/COP.filt3.sub/)",
    )
    parser.add_argument(
        "-t",
        "--test_file",
        type=str,
        default="",
        help="Path for the test json file (default None)",
    )
    parser.add_argument(
        "-f",
        "--to_filter",
        action="store_true",
        help="Filter labels from the text or not",
    )
    args = parser.parse_args()
    return args


def read_articles(folder_name):
    """Read articles from the json files and store them in separate lists for train, dev, and test"""
    train_articles, dev_articles, test_articles = [], [], []
    for filename in os.listdir(folder_name):
        with open(os.path.join(folder_name, filename), "r") as f:
            json_data = json.loads(f.read())
            if "23" in filename:
                dev_articles.extend(json_data["articles"])
            elif "24" in filename:
                test_articles.extend(json_data["articles"])
            else:
                train_articles.extend(json_data["articles"])
    return train_articles, dev_articles, test_articles


def read_single_file(file_path):
    """Read articles from a single json file"""
    with open(os.path.join(file_path), "r") as f:
        json_data = json.loads(f.read())
        return json_data['articles']


def create_words_to_filter(labels):
    """Create a list of words similar to labels"""
    # Remove "&" sign from the labels and divide them into separate words
    filter_words = [part.replace("&", "") for label in labels for part in label.split()]

    # Remove stopwords from the list of words and empty spaces
    filter_words_normal = [word for word in filter_words if word and word.lower() not in stopwords.words("english")]

    # Add singular and plural forms of the words (set a word limit of 3 as it does change USA to US and we don't want to remove us)
    filter_words_singular = [singularize(word) for word in filter_words_normal if len(singularize(word)) > 3]
    filter_words_plural = [pluralize(word) for word in filter_words_normal if len(pluralize(word)) > 3]

    # Combine the singular and plural forms of the words
    filter_words = []
    filter_words.extend(filter_words_normal)
    filter_words.extend(filter_words_singular)
    filter_words.extend(filter_words_plural)
    filter_words = list(set(filter_words))

    return filter_words


def filter_text(text, filter_words, to_filter):
    """Filter text by removing the filter words, phone numbers, URLS, multiple * and -."""
    filtered_text = " ".join(
        [
            word
            for word in text.split()
            if not bool([word_ for word_ in filter_words if word_ in word or word_ in word.lower()])
        ]
    ) if to_filter else text
    filtered_text = re.sub(r"http\S+", "", filtered_text, flags=re.MULTILINE)
    filtered_text = re.sub(
        r"((1-\d{3}-\d{3}-\d{4})|(\(\d{3}\) \d{3}-\d{4})|(\d{3}-\d{3}-\d{4}))", "", filtered_text, flags=re.MULTILINE
    )
    filtered_text = re.sub("(-{2})", "", filtered_text, flags=re.MULTILINE)
    filtered_text = re.sub("(\*{2})", "", filtered_text, flags=re.MULTILINE)

    return filtered_text


def calculate_prob_of_subjects(subjects):
    label_article = {}
    total_percentage = 0
    for subject in subjects:
        try:
            previous_percentage = int(subject['percentage'])
            total_percentage += previous_percentage
        except:
            pass
        
    
    for subject in subjects:
        try:
            percentage = int(subject['percentage'])
            percent_of_total = round(percentage / total_percentage, 4)
            label_article.update({subject['name']: percent_of_total})
        except:
            pass

        


    return [label_article[subject] if subject in label_article else 0 for subject in subject_counter.keys()]
    


def extract_features_labels(articles, labels_, to_filter):
    """Extract features and labels from the articles"""
    features, labels = [], []
    for article in articles:
        subjects = article["classification"]["subject"]
        if subjects and subjects[0]["name"].lower() in labels_:
            classification = article['classification']
            newspaper = article["newspaper"]
            organization = (
                classification["organization"][0]["name"]
                if classification["organization"]
                else "none"
            )
            industry = (
                classification["industry"][0]["name"] if classification["industry"] else "none"
            )
            geographic = (
                classification["geographic"][0]["name"]
                if classification["geographic"]
                else "none"
            )
            features.append(
                {
                    "body": filter_text(article["body"].lower(), filter_words, to_filter),
                    "headline": filter_text(article["headline"], filter_words, to_filter),
                    "newspaper": newspaper,
                    "organization": organization,
                    "all_subjects": calculate_prob_of_subjects([subject
                        for subject in subjects
                        if subject["name"] in labels_selected]),
                    "industry": industry,
                    "geographic": geographic,
                }
            )

            labels.append(subjects[0]["name"].lower())

    return features, labels


def store_data(features, labels, filename):
    """Store the features and labels in a json file"""
    with open(filename, "w") as f:
        json.dump({"features": features, "labels": labels}, f)


if __name__ == "__main__":
    args = create_arg_parser()
    train_articles, dev_articles, test_articles = read_articles(args.input_dir)
    labels = [
        article["classification"]["subject"][0]["name"].lower()
        for article in chain(train_articles, dev_articles, test_articles)
        if article["classification"]["subject"]
    ]
    threshold = 50
    labels_selected = [article for article, count in Counter(labels).most_common() if count > threshold]

    filter_words = create_words_to_filter(labels_selected)
    subject_counter = Counter()
    for article in chain(train_articles, dev_articles, test_articles):
        subjects = article['classification']['subject']
        if subjects:
            for subject in subjects:
                subject_counter[subject['name']] += 1

    if args.test_file:
        test_articles = read_single_file(args.test_file)

    X_train, Y_train = extract_features_labels(train_articles, labels_selected, args.to_filter)
    X_dev, Y_dev = extract_features_labels(dev_articles, labels_selected, args.to_filter)
    X_test, Y_test = extract_features_labels(test_articles, labels_selected, args.to_filter)


    store_data(X_train, Y_train, "./data/train.json")
    store_data(X_dev, Y_dev, "./data/dev.json")
    store_data(X_test, Y_test, "./data/test.json")
