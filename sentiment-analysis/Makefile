# Author:
#   Kian Ho <hui.kian.ho@gmail.com>
#
# Description:
#   This Makefile contains targets to generate the appropriate datasets; and to
#   perform training and evaluation of machine learning models for sentiment
#   analysis using Apache Spark and Python.
#
# Usage:
#   ...
# 
include ../Makefile.in

# Run the proof-of-concept pyspark experiment.
polarity2.0: data/polarity2.0/polarity2.0.csv
	$(SPARK_SUBMIT) ./run.py -i $<

# Make the master polarity2.0.csv file.
data/polarity2.0/polarity2.0.csv: data/polarity2.0/txt_sentoken
	ls -1 $</neg/* | ./make_train_data.py -l 0 > $@ 
	ls -1 $</pos/* | ./make_train_data.py -l 1 >> $@ 

clean:
	rm -rf data/polarity2.0/polarity2.0.csv

clean_raw:
