# -*- coding: utf-8 -*-
#
# Project: HORAE
# Author: Amir HAZEM
# Created: 24/09/2019
# Updated: 01/02/2021
#
# Role: Books of hours segmentation
# Generates line predictions and segmentation
# usig: SVM, BERT, Logit, naive_bayes...)


# Libraries
from __future__ import division
import codecs
import sys
import segmentation as ho
from os import listdir
from os.path import isfile, join


if __name__ == '__main__':

    # Inputs
    corpus_train = "All"  # all files in data/train/csv/ are automatically put All.csv
    corpus_test = ""  # all files in data/test/csv/
    corpus_id = '6dcf7706-f63d-4ede-9ffc-55b4f35faf2d'

    args = ho.load_args()
    data_type = str(args.data_type)  # "hier"  # str(sys.argv[3])  # flat / hier
    classifier = str(args.classifier)  # svm / bert / bert2 / logit / rf / dt / ada / mlp / xgb
    level = str(args.level)  # level1 / level12 / level123 / level2 / level3
    bool_train = eval(args.bool_train)  # True / False
    relaxation = int(args.relaxation)  # between 50 and 100
    # validation: evaluate segmentation with pk and windowdiff measures
    validation = eval(args.valid)  # True / False (possible only if test annotations are available)
    send_annotations = eval(args.send)  # True / False

    ho.printargs(args)
    # Data files
    directory = "../data/segmentation"
    path_train = directory + "/train/csv/" + str(data_type) + "/" + corpus_train + ".csv"
    path_test = directory + "/test/csv/" + str(data_type) + "/"

    path_pred = directory + "/test/pred/"
    path_seg = directory + "/test/seg/"
    # save model
    model_name = classifier + "_" + corpus_train + "_" + level
    path_save_model = directory + "/train/ML_models/" + model_name + '.sav'

    # for segmentation
    path_raw_seg = directory + "/raw/"

    try:
        # clean prediction and path segmentation
        ho.rm_dir(path_pred)
        ho.rm_dir(path_seg)
        print("Segmentation Processing... ")
        if classifier == "svm":
            ho.svm_line_classification(directory, path_train, bool_train, path_test,
                                       path_pred, path_save_model, classifier, level,
                                       relaxation, data_type, validation, send_annotations,
                                       corpus_id)
        else:
            if classifier == "bert" or classifier == "bert2":
                if classifier == "bert":
                    bert_type = "single"
                else:
                    bert_type = "pair"

                path_train = directory + "/train/csv/" + str(data_type) + "/bert/train_bert_"
                path_train += bert_type + '_' + level + ".csv"
                path_index = directory + "/train/csv/" + str(data_type) + "/bert/class_index_"
                path_index += str(level) + ".txt"

                ho.bert_line_classification(directory, path_train, bool_train, path_test,
                                            path_pred, path_index, classifier, level,
                                            relaxation, data_type, validation, send_annotations,
                                            path_index, corpus_id)
            else:
                print("Wrong classifier...")
    except Exception as exp:
        print("Line Classification Error ", sys.exc_info()[0])
        print(str(exp))
