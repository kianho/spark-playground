#!/usr/bin/env python
# encoding: utf-8
"""

Author:
    Kian Ho <hui.kian.ho@gmail.com>

Description:
    ...

Usage:
    run.py -i CSV

Options:
    -i, --input CSV     The input csv file.

References:
    - https://genomegeek.wordpress.com/2015/03/01/text-classification-in-apache-spark-1-2-1-using-python/

"""

import os
import sys
import re
import csv
import cStringIO

from docopt import docopt
from nltk.tokenize import word_tokenize

from pyspark import SparkConf, SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.classification import LabeledPoint

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
    return [ w for w in word_tokenize(text) if not NON_TOKEN_RE.search(w) ]
 

if __name__ == '__main__':
    opts = docopt(__doc__)
    
    conf = SparkConf().setMaster("local").setAppName("mrsparkle")
    sc = SparkContext(conf=conf)

    # Load the "raw" training data into RDDs.
    raw_rdd = sc.textFile(opts["--input"]).map(
            lambda line : load_record(line, quoting=csv.QUOTE_MINIMAL,
                fieldnames=["label", "text"]))

    hashing_tf = HashingTF()
    #idf = IDF()

    # Wrangling
    #
    # - Convert "label" column to boolean.   
    # - Tokenize and strip non-word tokens.
    train_rdd = raw_rdd.map(
        lambda rec : LabeledPoint(int(rec["label"]),
                                  hashing_tf.transform(tokenize(rec["text"]))))

    print train_rdd.take(5)
