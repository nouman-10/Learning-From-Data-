import argparse
import json
import os
from collections import Counter
from itertools import product
from operator import mul

import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus.reader.chasen import test
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

nltk.download("punkt")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
import argparse
import json
import random as python_random

import numpy as np
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import FunctionTransformer, LabelBinarizer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight

# Make reproducible as much as possible
np.random.seed(1234)
tf.random.set_seed(1234)
python_random.seed(1234)


def read_file(filename):
    with open(filename, "r") as f:
        data = json.loads(f.read())
    return data


def identity(x):
    return x


def preprocessor(X):
    """Preprocessor for the vectorizier to pre-process the data before fitting the model"""

    def penn2morphy(penntag):
        """Converts Penn Treebank tags to WordNet."""
        morphy_tag = {"NN": "n", "JJ": "a", "VB": "v", "RB": "r"}
        try:
            return morphy_tag[penntag[:2]]
        except:
            return "n"  # if mapping isn't found, fall back to Noun.

    # Initialize the lemmatizer
    wnl = WordNetLemmatizer()
    # Remove punctuation marks, do lemmatization and stemming, and remove stopwords

    processed_X = word_tokenize(X)
    processed_X = [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(processed_X)]

    return " ".join(processed_X)


baseline_model_types = {
    "multinomial": MultinomialNB,
    "svm": SVC,
    "knn": KNeighborsClassifier,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
}

vectorizer_types = {"count": CountVectorizer, "tfidf": TfidfVectorizer}


def print_scores(scores, score_name, labels):
    """Print the metrics along with their class"""
    print(f"{score_name} by class")
    for score, label in zip(scores, labels):
        print(f"{label}: {score:.3f}", end=" | ")

    print("\n")


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--data_dir",
        default="./data/",
        type=str,
        help="Input directory to read json files from (default ./data/)",
    )
    args = parser.parse_args()
    return args


def read_embeddings(embeddings_file):
    """Read in word embeddings from file and save as numpy array"""
    embeddings = open(embeddings_file, "r").read()
    return {line.split()[0]: np.array(line.split()[1:], dtype=float) for line in embeddings.split("\n")[:-1]}


def get_emb_matrix(voc, emb):
    """Get embedding matrix given vocab and the embeddings"""
    num_tokens = len(voc) + 2
    word_index = dict(zip(voc, range(len(voc))))
    # Bit hacky, get embedding dimension from the word "the"
    embedding_dim = len(emb["the"])
    # Prepare embedding matrix to the correct size
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = emb.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    # Final matrix with pretrained embeddings that we can feed to embedding layer
    return embedding_matrix


def test_set_predict(model, X_test, Y_test, ident="Dev"):
    """Do predictions and measure accuracy on our own test set (that we split off train)"""
    # Get predictions using the trained model
    Y_pred = model.predict(X_test)
    # Finally, convert to numerical labels to get scores with sklearn
    Y_pred = np.argmax(Y_pred, axis=1)
    # If you have gold data, you can calculate accuracy
    Y_test = np.argmax(Y_test, axis=1)
    print("Accuracy on own {1} set: {0}".format(round(accuracy_score(Y_test, Y_pred), 3), ident))


def upsample(X_train, Y_train, size=100, downsample=False):
    label_count = dict(Counter(Y_train).most_common())
    new_X_train, new_Y_train = [], []

    for label, count in label_count.items():
        label_ids = [i for i, y in enumerate(Y_train) if y == label]
        desired_count = count
        x = np.array(X_train)[label_ids]
        y = np.array(Y_train)[label_ids]
        x, y = shuffle(x, y, random_state=123)

        if desired_count > size and downsample:
            new_X_train.extend(x[:size])
            new_Y_train.extend(y[:size])
        else:
            new_X_train.extend(x)
            new_Y_train.extend(y)
        while desired_count < size:
            new_X_train.extend(x)
            new_Y_train.extend(y)

            desired_count += count

    new_X_train, new_Y_train = shuffle(new_X_train, new_Y_train, random_state=123)

    return new_X_train, new_Y_train


class BaselineModel:
    def __init__(self, data_path, model_type, model_params, vectorizer_type, vectorizer_params, extra_feature=None):
        self.data_path = data_path
        self.model = baseline_model_types.get(model_type)
        self.vectorizer = vectorizer_types.get(vectorizer_type)
        if self.vectorizer is None or self.model is None:
            raise Exception("Invalid vectorizer or model type")
        self.model_params = model_params
        self.vectorizer_params = vectorizer_params
        self.extra_feature = extra_feature
        self.X_train = None
        self.X_test = None
        self.X_dev = None
        self.Y_dev = None
        self.Y_train = None
        self.Y_test = None
        self.labels = None
        self.predictions = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.confusion_matrix = None

    def read_data(self):
        train_data = read_file(os.path.join(self.data_path, "train.json"))
        dev_data = read_file(os.path.join(self.data_path, "dev.json"))
        test_data = read_file(os.path.join(self.data_path, "test.json"))

        if self.extra_feature:
            self.X_train, self.Y_train = train_data["features"], train_data["labels"]
            self.X_dev, self.Y_dev = dev_data["features"], dev_data["labels"]
            self.X_test, self.Y_test = test_data["features"], test_data["labels"]
        else:
            self.X_train, self.Y_train = [article["body"] for article in train_data["features"]], train_data["labels"]
            self.X_dev, self.Y_dev = [article["body"] for article in dev_data["features"]], dev_data["labels"]
            self.X_test, self.Y_test = [article["body"] for article in test_data["features"]], test_data["labels"]
            if upsample:
                self.X_train, self.Y_train = upsample(self.X_train, self.Y_train)

        self.labels = list(set(self.Y_train))

    def train_baseline(self):
        if self.extra_feature:
            self.train_with_extra_feature()
        else:
            self.pipeline = Pipeline(
                [("vect", self.vectorizer(**self.vectorizer_params)), ("clf", self.model(**self.model_params))]
            )
            self.pipeline.fit(self.X_train, self.Y_train)        

    def evaluate(self):
        self.predictions = self.pipeline.predict(self.X_test)
        self.accuracy = accuracy_score(self.Y_test, self.predictions)
        self.precision = precision_score(self.Y_test, self.predictions, average=None)
        self.recall = recall_score(self.Y_test, self.predictions, average=None)
        self.f1 = f1_score(self.Y_test, self.predictions, average=None)
        self.confusion_matrix = confusion_matrix(self.Y_test, self.predictions)

    def train_with_extra_feature(self):
        print(self.extra_feature)

        def extra_feature_body_extractor(articles):
            # construct object dtype array with two columns
            # first column = 'body' and second column = 'extra'
            features = np.empty(shape=(len(articles), 2), dtype=object)
            for i, article in enumerate(articles):

                extra_feature, body = article[self.feature], article["body"]
                # store body text in second column
                features[i, 0] = body
                features[i, 1] = extra_feature

            return features

        extra_feature_body_transformer = FunctionTransformer(extra_feature_body_extractor)
        self.pipeline = Pipeline(
            [
                # Extract subject & body
                ("extra_feature_body", extra_feature_body_transformer),
                # Use ColumnTransformer to combine the extra and body features
                (
                    "union",
                    ColumnTransformer(
                        [
                            # Tfidf for body
                            ("body", self.vectorizer(**self.vectorizer_params), 0),
                            # Tfidf for extra feature
                            ("extra_feature", TfidfVectorizer(preprocessor=identity, tokenizer=word_tokenize), 1),
                        ]
                    ),
                ),
                # Use a SVC classifier on the combined features
                ("clf", self.model(**self.model_params)),
            ]
        )
        self.pipeline.fit(self.X_train, self.Y_train)

    def train_with_all_baseline_combinations(self):
        model_names = [
            ("multinomial", {}),
            ("svm", {"kernel": ["rbf", "linear"], "C": [0.1, 0.5, 1.0]}),
            ("knn", {"k": [1, 5, 9, 13]}),
            ("decision_tree", {}),
            ("random_forest", {"n_estimators": [10, 100, 200, 500]}),
        ]

        vectorizer_parameters = {
            "preprocessor": [identity, preprocessor],
            "tokenizer": [word_tokenize],
            "max_features": [None, 10000, 50000],
            "max_df": [0.95, None],
            "min_df": [0.01, None],
            "ngram_range": [(1, 1), (1, 2), (1, 3)],
        }

        all_combinations = [
            dict(zip(vectorizer_parameters.keys(), params)) for params in product(*vectorizer_parameters.values())
        ]

        # Test all combinations of models and vectorizers
        for model_type, model_params in model_names:
            for vectorizer_type in ["count", "tfidf"]:
                for combination in all_combinations:
                    self.model = baseline_model_types.get(model_type)
                    self.vectorizer = vectorizer_types.get(vectorizer_type)
                    self.model_params = model_params
                    self.vectorizer_params = combination
                    self.train_baseline()
                    self.evaluate()
                    self.print_results()
                    self.save_results(os.path.join(args.output_path, f"{model_type}_{combination}.txt"))

    def train_for_additional_features(self):
        # Test all additional features
        additional_features = ["industry", "organization", "geographic", "newspaper", "headline"]

        self.model = baseline_model_types.get("svm")
        self.vectorizer = vectorizer_types.get("tfidf")
        self.model_params = {"kernel": "linear"}
        self.vectorizer_params = {"max_df": 0.95, "max_features": 50000, "ngram_range": (1, 2)}

        for feature in additional_features:
            self.extra_feature = feature

            self.read_data()
            self.train_baseline()
            self.evaluate()
            self.print_results()
            self.save_results(os.path.join(args.output_path, f"baseline_{feature}.txt"))

    def build_and_compile_model(self, combination, num_tokens, num_labels, emb_matrix):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Embedding(
                num_tokens,
                combination["glove_length"],
                embeddings_initializer=tf.keras.initializers.Constant(emb_matrix),
                trainable=combination["glove_trainable"],
            )
        )
        # Here you should add LSTM layers (and potentially dropout)

        multiple_lstm_layers = combination["no_of_lstm"] > 1
        model.add(tf.keras.layers.LSTM(units=512, activation=combination["activation_function"], return_sequences=multiple_lstm_layers))
        model.add(tf.keras.layers.Dropout(combination["dropout"]))

        if multiple_lstm_layers:
            count = 1
            while count < multiple_lstm_layers:
                model.add(
                    tf.keras.layers.LSTM(
                        units=512, activation=combination["activation_function"], return_sequences=multiple_lstm_layers
                    )
                )
                model.add(tf.keras.layers.Dropout(combination["dropout"]))
                count += 1

            model.add(tf.keras.layers.LSTM(units=512, activation=combination["activation_function"]))
            model.add(tf.keras.layers.Dropout(combination["dropout"]))

        multiple_dense_layers = combination["no_of_dense"] > 1

        if multiple_dense_layers:
            model.add(tf.keras.layers.Dense(input_dim=combination["glove_length"], units=512, activation="relu"))
            model.add(tf.keras.layers.Dense(units=num_labels, activation="softmax"))
        else:
            model.add(tf.keras.layers.Dense(input_dim=combination["glove_length"], units=num_labels, activation="softmax"))

        model.compile(
            loss=combination["loss_function"],
            optimizer=combination["optimizer"](learning_rate=combination["lr"]),
            metrics=["accuracy"],
        )

        return model

    def get_callbacks(epochs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=int(epochs // 5))
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=int(epochs // 8),
            verbose=0,
            mode="auto",
            min_delta=0.001,
            cooldown=0,
            min_lr=0.00000001,
        )

        return [early_stopping, reduce_lr]

    def train_lstm_combinations(self):
        lstm_parameters = {
            "output_length": [50, 100, None],
            "lr": [0.01, 0.001, 0.0001],
            "loss_function": ["categorical_cross_entropy", "sparsecategorical_cross_entropy"],
            "glove_length": [50, 100, 200, 300],
            "glove_trainable": [True, False],
            "dropout": [0.1, 0.3, 0.5],
            "units": [128, 256, 512],
            "no_of_lstm": [1, 2, 3],
            "activation": ["relu", "tanh"],
            "no_of_dense": [1, 2],
            "batch_size": [8, 16, 32],
            "epochs": [25, 50, 100],
            "optimizer": [tf.keras.optimizers.Adam, tf.keras.optimizers.SGD, tf.keras.optimizers.RMSprop],
        }

        all_lstm_combinations = [
            dict(zip(lstm_parameters.keys(), params)) for params in product(*lstm_parameters.values())
        ]

        for combination in all_lstm_combinations:
            embeddings = read_embeddings(f"./data/glove.6B.{combination['glove_length']}d.txt")

            vectorizer = tf.keras.layers.TextVectorization(standardize=None, output_sequence_length=combination["output_length"])
            text_ds = tf.data.Dataset.from_tensor_slices(self.X_train + self.X_dev)
            vectorizer.adapt(text_ds)
            voc = vectorizer.get_vocabulary()
            emb_matrix = get_emb_matrix(voc, embeddings)
            X_train_vect = vectorizer(np.array([[s] for s in self.X_train])).numpy()
            X_dev_vect = vectorizer(np.array([[s] for s in self.X_dev])).numpy()

            num_tokens = len(emb_matrix)
            num_labels = len(self.Y_train[0])

            encoder = LabelBinarizer()
            Y_train_bin = encoder.fit_transform(self.Y_train)  # Use encoder.classes_ to find mapping back
            Y_dev_bin = encoder.transform(self.Y_dev)

            y_integers = np.argmax(self.Y_train, axis=1)
            class_weights = compute_class_weight("balanced", np.unique(y_integers), y_integers)
            d_class_weights = dict(enumerate(class_weights))

            model = self.build_and_compile_model(combination, num_tokens, num_labels)

            callbacks = self.get_callbacks(combination["epochs"])

            model.fit(
                X_train_vect,
                Y_train_bin,
                batch_size=combination["batch_size"],
                epochs=combination["epochs"],
                validation_data=(X_dev_vect, Y_dev_bin),
                callbacks=callbacks,
                class_weight=d_class_weights,
                verbose=0,
            )

            print("Result for: ", combination)
            test_set_predict(model, X_dev_vect, Y_dev_bin)

    def print_results(self):
        print(f"Accuracy: {self.accuracy:.3f}")
        print_scores(self.precision, "Precision", self.labels)
        print_scores(self.recall, "Recall", self.labels)
        print_scores(self.f1, "F1", self.labels)
        print(self.confusion_matrix)

    def save_results(self, filename):
        with open(filename, "w") as f:
            f.write("Accuracy: " + str(self.accuracy) + "\n")
            f.write("Precision: " + str(self.precision) + "\n")
            f.write("Recall: " + str(self.recall) + "\n")
            f.write("F1: " + str(self.f1) + "\n")
            f.write("Confusion Matrix: \n")
            f.write(str(self.confusion_matrix))


if __name__ == "__main__":
    args = create_arg_parser()

    # Best Model
    model = BaselineModel(
        data_path=args.data_dir,
        model_type="svm",
        model_params={"kernel": "linear"},
        vectorizer_type="tfidf",
        vectorizer_params={"max_df": 0.95, "max_features": 50000, "ngram_range": (1, 2)},
        extra_feature=None,
    )
    model.read_data()
    model.train_baseline()
    model.evaluate()
    model.print_results()
    model.save_results("best_baseline_results.txt")
