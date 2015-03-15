#!/usr/bin/env python
# encoding: utf-8
"""

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    A script that explores some of the functionality of PySpark and its `mllib`
    machine learning library.

Usage:
    run.py -i CSV

Options:
    -i, --input CSV     The input csv file.

References:
    - http://spark.apache.org/docs/1.3.0/api/python/index.html
    - https://genomegeek.wordpress.com/2015/03/01/text-classification-in-apache-spark-1-2-1-using-python/

"""

import os
import sys
import re
import csv
import cStringIO
import sklearn.metrics

from docopt import docopt
from nltk.tokenize import word_tokenize

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.ml.feature import HashingTF
from pyspark.mllib.feature import Word2Vec
from pyspark.mllib.classification import LabeledPoint, SVMWithSGD


# Tokens that match this regular expression will be ignored.
NON_TOKEN_RE = re.compile(r"""
    # Puncutation
    (^[!"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~]+$)
    |
    # Other non-word characters, taken from nltk.tokenize.punkt.
    (^((?:[?!)\";}\]\*:@\'\({\[]))$)
    |
    # Multi-character punctuation, e.g. ellipses, en-, and em-dashes.
    (^(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)$)
    |
    # Integer or floating-point number.
    (\d+\.?\d*")
    |
    # Single non-alphabetic character.
    (^[^a-zA-Z]$)
""", re.VERBOSE)


def load_record(line, **reader_kwargs):
    """Parse a single line in a csv file.

    """

    f = cStringIO.StringIO(line)
    reader = csv.DictReader(f, **reader_kwargs)
    return reader.next()


def tokenize(text):
    return [ w.lower() for w in word_tokenize(text) if not NON_TOKEN_RE.search(w) ]


if __name__ == '__main__':
    opts = docopt(__doc__)
    
    conf = SparkConf().setMaster("local").setAppName("mrsparkle")
    sc = SparkContext(conf=conf)

    # Load the "raw" training data into RDDs.
    # Could probably use Spark SQL to load csvs directly into RDDs.
    raw_rdd = sc.textFile(opts["--input"]).map(
            lambda line : load_record(line, quoting=csv.QUOTE_MINIMAL,
                fieldnames=["label", "text"]))

    # Experimenting with spark Rows.            
    rows_rdd = raw_rdd.map(
            lambda rec: Row(label=int(rec["label"]),
                            tokens=tokenize(rec["text"])))

    # Experimenting with spark DataFrames.                
    # Need to create an `SQLContext` from `sc` in order to use the spark
    # dataframe API.
    sc2 = SQLContext(sc)
    ddf = sc2.createDataFrame(rows_rdd)

    # Use the "hashing trick" to convert each token list to a vector of term
    # frequencies.
    htf = HashingTF(numFeatures=10000, inputCol="tokens", outputCol="features")
    tf_ddf = htf.transform(ddf)

    # Convert to RDD of LabeledPoints (compatible with mllib classifiers).
    # NOTE: `select(...)` wil return an RDD of rows.
    X = tf_ddf.select("*").map(
            lambda r : LabeledPoint(int(r.label), r.features))

    X_pos = X.filter(lambda x : x.label == 1)
    X_neg = X.filter(lambda x : x.label == 0)

    # Use 30% of the instances as a held-out test set, stratified.
    X_pos_train, X_pos_test = X_pos.randomSplit([0.7, 0.3])
    X_neg_train, X_neg_test = X_neg.randomSplit([0.7, 0.3])

    X_train = X_pos_train.union(X_neg_train)
    X_test = X_pos_test.union(X_neg_test)

    # Use default SVM parameters for now (linear kernel w/ L2-regularization)
    SVM = SVMWithSGD.train(X_train)

    # Do the predictions, store results as <predicted>,<true> tuples.
    results = X_test.map(lambda x : (SVM.predict(x.features), x.label))

    y_pred = results.map(lambda x : x[0]).collect()
    y_true = results.map(lambda x : x[1]).collect()

    # Compute + display accuracy, F1, precision, recall, and MCC
    acc = sklearn.metrics.accuracy_score(y_true, y_pred)
    f1 = sklearn.metrics.f1_score(y_true, y_pred)
    prec = sklearn.metrics.precision_score(y_true, y_pred)
    recall = sklearn.metrics.recall_score(y_true, y_pred)
    mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)

    print("Acc:{acc:.2f} F1:{f1:.2f} Prec:{prec:.2f} "
          "Rec:{recall:.2f} MCC:{mcc:.2f}".format(acc=acc, f1=f1, prec=prec,
              recall=recall, mcc=mcc))

    sys.exit()

    w2v = Word2Vec()
    w2v_model = w2v.fit(ddf.select("tokens"))

    sys.exit()

    # Find synonums.
    synonyms = w2v_model.findSynonyms("good", 50)

    for word, cos_sim in synonyms:
        print "{}: {}".format(word, cos_sim)
