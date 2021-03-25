# -*- coding: utf-8 -*-
#
#
# Project: HORAE
# Author: Amir HAZEM
# Created: 24/09/2019
# Updated: 25/03/2021
#
#
# Role:
# Preprocesses raw books of hours and generate train/test data
# This script generates train and test csv files:
# train --> "../data/segmentation/train/csv/hier/"
# test  --> "../data/segmentation/train/csv/flat/"
# flat: do not consider hierarchie between sections of levels 1 to 3
# hier: considers hierarchie in the tagset
#       level2 for instance is the concatenation of
#       tagset of level1 and level2 and is represented as level12


# Libraries
from __future__ import division
import sys
import segmentation as ho


if __name__ == '__main__':

    # Inputs
    directory = "../data/segmentation"
    path_in_train_raw = directory + "/train/raw/"
    path_in_test_raw = directory + "/test/raw/"
    path_class_seg = "../data/segmentation/annotation_class/"
    # Outputs
    path_out_train_csv_hier = directory + "/train/csv/hier/"
    path_out_train_csv_flat = directory + "/train/csv/flat/"

    # Hierarchical classes are:
    # level1 level12 level123 level23
    path_out_test_csv_hier = directory + "/test/csv/hier/"

    # Flat classes are:
    # level1 level2 level3
    path_out_test_csv_flat = directory + "/test/csv/flat/"

    # choi format
    path_out_flat_choi = directory + "/test/choiformat/flat/"
    path_out_hier_choi = directory + "/test/choiformat/hier/"
    path_out_textonly = directory + "/test/txt/"

    path_train_ml_models = directory + "/train/ML_models/"
    path_test_seg = directory + "/test/seg/"
    path_test_pred = directory + "/test/pred/"
    path_test_comp = directory + "/test/comp/"

    try:
        # Clean output directories
        ho.rm_dir(path_out_train_csv_hier)
        ho.rm_dir(path_out_train_csv_flat)
        ho.rm_dir(path_out_test_csv_hier)
        ho.rm_dir(path_out_test_csv_flat)
        ho.rm_dir(path_out_flat_choi)
        ho.rm_dir(path_out_hier_choi)
        ho.rm_dir(path_out_textonly)

        ho.rm_dir(path_train_ml_models)
        ho.rm_dir(path_test_seg)
        ho.rm_dir(path_test_seg)
        ho.rm_dir(path_test_seg)

        ho.rm_dir(path_out_train_csv_hier + '/bert/')
        ho.rm_dir(path_out_train_csv_flat + '/bert/')
        ho.rm_dir(path_out_test_csv_hier + '/bert/')
        ho.rm_dir(path_out_test_csv_flat + '/bert/')

        ho.generate_choi_txt(path_in_test_raw, path_out_flat_choi,
                             path_out_hier_choi, path_out_textonly)

        print("Generate flat and hierarchical training data...")
        (tab_sec1,
         tab_sec12,
         tab_sec123,
         tab_sec2,
         tab_sec3) = ho.generate_train_ML_transcriptions(path_in_train_raw,
                                                         path_out_train_csv_flat,
                                                         path_out_train_csv_hier)

        print("Generate flat and hierarchical test data...")
        ho.generate_test_ML_transcriptions(path_in_test_raw,
                                           path_out_test_csv_flat,
                                           path_out_test_csv_hier)

        print("Generate Train/Test data for BERT...")
        ho.generate_train_test_BERT(path_out_train_csv_hier,
                                    path_out_test_csv_hier,
                                    path_out_train_csv_flat,
                                    path_out_test_csv_flat,
                                    tab_sec1, tab_sec12,
                                    tab_sec123, tab_sec2,
                                    tab_sec3)

    except Exception as exp:

        print("Unexpected error ", sys.exc_info()[0])
        print(str(exp))
