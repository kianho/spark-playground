from pyspark import SparkConf, SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF, IDF

conf = SparkConf().setMaster("local").setAppName("foo-bar")
sc = SparkContext(conf=conf)

# list of (<label>,<text>) tuples
docs = [(1, "Hello hello world dog"), (2, "Hello WORLD world cat"), 
        (3, "cat dog")]
rdd = sc.parallelize(docs)

htf = HashingTF(1000)
idf = IDF()

# keep the label, tokenize, then compute the term-frequency hashes.
# convert the result to collection of LabelPoints.
toks_rdd = rdd.map(
    lambda (label, text):
        LabeledPoint(label, htf.transform(text.lower().split())))

idf.fit(toks_rdd).transform(toks_rdd)

labels_before = rdd.map(lambda (label, text): label).collect()
labels_after = toks_rdd.map(lambda x: int(x.label)).collect()

# labels_before == labels_after guaranteed ?
