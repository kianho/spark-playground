from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF

conf = SparkConf().setMaster("local").setAppName("foo-bar")
sc = SparkContext(conf=conf)

# list of (<label>,<text>) tuples
docs = [(1, "Hello hello world dog"), (2, "Hello WORLD world cat"), 
        (3, "cat dog")]
rdd = sc.parallelize(docs, 2)

htf = HashingTF(1000)
idf = IDF()

# keep the label, tokenize, then compute the term-frequency hashes.
toks_rdd = rdd.map(
    lambda (label, text):
        LabeledPoint(label, htf.transform(text.lower().split())))
labels_rdd = toks_rdd.map(lambda x: x.label)

# keep only the TF vectors:
tfvec_rdd = toks_rdd.map(lambda x: x.features) 
tfidf_rdd = idf.fit(tfvec_rdd).transform(tfvec_rdd)

# Is the original ordering of entries in `toks_rdd` preserved in `tdfidf_rdd`?
print toks_rdd.map(lambda x: x[1]).collect()
print labels_rdd.zip(tfidf_rdd).collect()
