# -*- coding: utf-8 -*-
#
#
# Project: HORAE
# Author: Amir HAZEM
# Created: 24/09/2019
# Updated: 02/12/2020
#
#
# Role: Line classification (ML)
#
# Generates line predictions of a given machine learning classifier
# that is: SVM, Logit, naive_bayes...)


# Libraries
from __future__ import division
import codecs
import sys
import segmentation as ho
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':

    # Inputs
    corpus_train = "All"  # sys.argv[1] All / or corpus name in data/train/csv/...
    # Arsenal651 for instance
    corpus_test = "" #  sys.argv[2]
    data_type = "hier"  # str(sys.argv[3])  # flat / hier
    classifier = "svm"  # sys.argv[4]  # svm / logit / gnb / rf / dt / ada / mlp / xgb
    level = "level1"  # sys.argv[5]  # level1 / level2 / level3 (hierarchical 1 + 2 + 3)
    bool_train = True  # eval(sys.argv[6])
    relaxation = 50  # int(sys.argv[7])  # 50 / 100 / ...
    # bool_train = False
    directory = "../data/segmentation"
    path_train = directory + "/train/csv/" + str(data_type) + "/" + corpus_train + ".csv"
    path_test = directory + "/test/csv/" + str(data_type) + "/"

    path_pred = directory + "/test/pred/"
    # save model
    model_name = classifier + "_" + corpus_train + "_" + level
    path_save_model = directory + "/train/ML_models/" + model_name + '.sav'

    # for segmentation
    path_raw_seg = directory + "/raw/"

    try:
        print("Line Classification Processing... ")
        ho.gen_line_classification(directory, path_train, bool_train, path_test,
                                   path_pred, path_save_model, classifier, level,
                                   relaxation, data_type)

    except Exception as exp:
        print("Line Classification Error ", sys.exc_info()[0])
        print(str(exp))
