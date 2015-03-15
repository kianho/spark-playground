# spark-playground

## Installation (Ubuntu)

1. Download and install Python 2.7.x using one of the following methods:
  - `sudo apt-get install python2.7`
  - [Anaconda Python Distribution](download-anaconda)
  - [https://www.python.org/downloads/](download-python)

2. Install the requisite Python libraries:
  - `pip install scikit-learn pandas docopt`

3. Download and install Apache Spark
  - https://spark.apache.org/downloads.html.

4. Clone this repository:
  - `git clone git@github.com:kianho/spark-playground.git`

5. Change the `SPARK_HOME` variable in [`Makefile.in`](Makefile.in) to the install
   directory used in step 3.

## Example usage

Experiment with sentiment analysis, run the following:
```
cd ./spark-playground/sentiment-analysis/
make polarity2.0
```
which will:

1. wrangle/parse the polarity2.0 dataset into Spark RDDs.
2. partition the dataset into a training and held-out test set.
3. train a linear SVM with default parameters.
4. compute and display performance measures to stdout.


[download-spark]: https://spark.apache.org/downloads.html
[download-anaconda]: http://continuum.io/downloads
