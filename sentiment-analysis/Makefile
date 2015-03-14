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
export SHELL
SHELL:=/bin/bash
.SECONDARY:

# Change this according to your own spark installation.
SPARK_SUBMIT:=~/Downloads/spark-1.3.0-bin-hadoop2.4/bin/spark-submit


all: data/polarity2.0/polarity2.0.csv
	$(SPARK_SUBMIT) ./run.py -i $<

# Make the master polarity2.0.csv file.
data/polarity2.0/polarity2.0.csv: data/polarity2.0/txt_sentoken
	ls -1 $</neg/* | ./make_train_data.py -l 0 > $@ 
	ls -1 $</pos/* | ./make_train_data.py -l 1 >> $@ 

clean:
	rm -rf data/polarity2.0/polarity2.0.csv

clean_raw:
