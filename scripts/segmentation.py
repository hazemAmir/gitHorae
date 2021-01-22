# -*- coding: utf-8 -*-
#
#
# Project: HORAE
# Author: Amir HAZEM
# Created: 24/09/2019
# Updated: 02/12/2020
#
# Role: HORAE Library

from os import listdir
from os.path import isfile, join, splitext
import codecs
import nltk
import sys
import pandas as pd
import pickle
# Import required libraries for machine learning classifiers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from nltk.metrics.segmentation import pk, windowdiff
import segeval as se
# Functions
#
#
# Preprocessing functions
#
# Clean text


def clean(text):

    line_ = (text.rstrip())
    sent_tok1 = (' '.join(nltk.word_tokenize(line_.lower())))
    sent_tok = ' '.join(x for x in sent_tok1.split(' ') if x.isalpha())

    return sent_tok


# extract_class
def extract_class(line):

    line = line.strip()
    ch = line.split('\t')
    text = ch[0]

    if len(ch) == 2:
        level1 = ch[1]
        level2 = "no_level2"
        level3 = "no_level3"
        level4 = "no_level4"

    if len(ch) == 3:
        level1 = ch[1]
        level2 = ch[2]
        level3 = "no_level3"
        level4 = "no_level4"

    if len(ch) == 4:
        level1 = ch[1]
        level2 = ch[2]
        level3 = ch[3]
        level4 = "no_level4"

    if len(ch) == 5:
        level1 = ch[1]
        level2 = ch[2]
        level3 = ch[3]
        level4 = ch[4]

    level12 = level1 + "_" + level2
    level123 = level1 + "_" + level2 + "_" + level3
    level1234 = level1 + "_" + level2 + "_" + level3 + "_" + level4
    level23 = level2 + "_" + level3
    level234 = level2 + "_" + level3 + "_" + level4
    level34 = level3 + "_" + level4

    text_clean = clean(text)
    string_flat = text_clean + '\t' + level1 + '\t' + level2 + '\t' + level3
    string_hier = text_clean + '\t' + level1 + '\t' + level12 + '\t' + level123
    return text_clean, string_flat, string_hier


# Generate train data for ML classification
def generate_train_ML_transcriptions(path_in, path_out_flat, path_out_hier):

    head_flat = "line_number" + '\t' + "line" + '\t' + "class_level1" +\
                '\t' + "class_level2" + '\t' + "class_level3"
    head_hier = "line_number" + '\t' + "line" + '\t' + "class_level1" +\
                '\t' + "class_level12" + '\t' + "class_level123"

    cpt_all = 1
    onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    with codecs.open(path_out_flat + '/' + "All.csv", 'w',
         encoding='utf-8') as ft, codecs.open(path_out_hier + '/' +
         "All.csv", 'w', encoding='utf-8') as ft2:

        ft.write(head_flat + "\n")
        ft2.write(head_hier + "\n")

        for filename in onlyfiles:

            print(filename)
            with codecs.open(path_out_flat + '/' + filename +
                 ".csv", 'w', encoding='utf-8') as fout, codecs.open(
                 path_out_hier + '/' + filename + ".csv",
                 'w', encoding='utf-8') as fout2:
                with codecs.open(path_in + '/' + filename, 'r',
                                 encoding='utf-8') as f:

                    fout.write(head_flat + "\n")
                    fout2.write(head_hier + "\n")
                    cpt = 1
                    # skip head
                    next(f)
                    for line in f:

                        clean_text, line_flat, line_hier = extract_class(line)

                        if len(clean_text.split(' ')) > 1:

                            fout.write(str(cpt) + '\t' + line_flat + '\n')
                            ft.write(str(cpt_all) + '\t' + line_flat + '\n')
                            fout2.write(str(cpt) + '\t' + line_hier + '\n')
                            ft2.write(str(cpt_all) + '\t' + line_hier + '\n')
                            cpt += 1
                            cpt_all += 1


# Generate test data for ML classification
def generate_test_ML_transcriptions(path_in, path_out_flat, path_out_hier):

    head_flat = "line_number" + '\t' + "line" + '\t' + "class_level1" +\
                '\t' + "class_level2" + '\t' + "class_level3"
    head_hier = "line_number" + '\t' + "line" + '\t' + "class_level1" +\
                '\t' + "class_level12" + '\t' + "class_level123"

    cpt_all = 1
    onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]
    with codecs.open(path_out_flat + '/' + "All.csv", 'w',
         encoding='utf-8') as ft:

        ft.write(head_flat + "\n")

        for filename in onlyfiles:

            print(filename)
            with codecs.open(path_out_flat + '/' + filename +
                 ".csv", 'w', encoding='utf-8') as fout, codecs.open(
                 path_out_hier + '/' + filename + ".csv",
                 'w', encoding='utf-8') as fout2:
                with codecs.open(path_in + '/' + filename, 'r',
                                 encoding='utf-8') as f:

                    fout.write(head_flat + "\n")
                    fout2.write(head_hier + "\n")
                    cpt = 1
                    # skip head
                    next(f)
                    for line in f:

                        clean_text, line_flat, line_hier = extract_class(line)

                        if len(clean_text.split(' ')) > 1:

                            fout.write(str(cpt) + '\t' + line_flat + '\n')
                            ft.write(str(cpt_all) + '\t' + line_flat + '\n')
                            fout2.write(str(cpt) + '\t' + line_hier + '\n')
                            cpt += 1
                            cpt_all += 1


# Generate reference and test data for segmentation
def generate_choi_txt(path_in, path_out_flat_choi, path_out_hier_choi,
                      path_out_textonly):

    onlyfiles = [f for f in listdir(path_in) if isfile(join(path_in, f))]

    for filename in onlyfiles:

        level_pred1 = ""
        level_pred2 = ""
        level_pred3 = ""
        level_pred12 = ""
        level_pred123 = ""

        print(filename)

        with codecs.open(path_out_flat_choi + '/' + filename + "_level1.ref",
             'w', encoding='utf-8') as foutflat1,\
            codecs.open(path_out_flat_choi + '/' + filename + "_level2.ref",
                        'w', encoding='utf-8') as foutflat2,\
            codecs.open(path_out_flat_choi + '/' + filename + "_level3.ref",
                        'w', encoding='utf-8') as foutflat3,\
            codecs.open(path_out_hier_choi + '/' + filename + "_level1.ref",
                        'w', encoding='utf-8') as fouthier1,\
            codecs.open(path_out_hier_choi + '/' + filename + "_level12.ref",
                        'w', encoding='utf-8') as fouthier12,\
            codecs.open(path_out_hier_choi + '/' + filename+"_level123.ref",
                        'w', encoding='utf-8') as fouthier123,\
            codecs.open(path_out_textonly + '/' + filename + ".txt",
                        'w', encoding='utf-8') as fout3:
            with codecs.open(path_in + '/' + filename, 'r',
                             encoding='utf-8') as f:

                cpt = 1
                # skip head
                next(f)
                for line in f:
                    clean_text, line_flat, line_hier = extract_class(line)

                    if len(clean_text.split(' ')) > 1:

                        flat_levels = line_flat.split('\t')
                        level1 = flat_levels[1]
                        level2 = flat_levels[2]
                        level3 = flat_levels[3]

                        hier_levels = line_hier.split('\t')

                        level12 = hier_levels[2]
                        level123 = hier_levels[3]

                        if level1 != level_pred1:
                            foutflat1.write("==========" + '\n')
                            fouthier1.write("==========" + '\n')
                            level_pred1 = level1

                        if level2 != level_pred2:
                            foutflat2.write("==========" + '\n')
                            level_pred2 = level2

                        if level12 != level_pred12:
                            fouthier12.write("==========" + '\n')
                            level_pred12 = level12

                        if level3 != level_pred3:
                            foutflat3.write("==========" + '\n')
                            level_pred3 = level3

                        if level123 != level_pred123:
                            fouthier123.write("==========" + '\n')
                            level_pred123 = level123

                        # write choi format for each level
                        foutflat1.write(clean_text + '\n')
                        fouthier1.write(clean_text + '\n')
                        foutflat2.write(clean_text + '\n')
                        foutflat3.write(clean_text + '\n')
                        fouthier12.write(clean_text + '\n')
                        fouthier123.write(clean_text + '\n')

                        # write txt line only
                        fout3.write(clean_text + '\n')
            # add a final line
            # foutflat1.write('\n')
            # fouthier1.write('\n')
            # foutflat2.write('\n')
            # foutflat3.write('\n')
            # fouthier12.write('\n')
            # fouthier123.write('\n')


# Machine learning functions

def train(path_train, path_save_model, level, classifier):
    # train
    df = pd.read_csv(path_train, delimiter='\t')
    col = ['class_' + level, 'line']

    print("---------  Train class size  --------")
    if level == "level1":
        print(df.class_level1.value_counts())
    if level == "level2":
        print(df.class_level2.value_counts())
    if level == "level3":
        print(df.class_level3.value_counts())
    if level == "level4":
        print(df.class_level4.value_counts())
    if level == "level12":
        print(df.class_level12.value_counts())
    if level == "level123":
        print(df.class_level123.value_counts())
    if level == "level1234":
        print(df.class_level1234.value_counts())
    print("-------------------------------------")

    df['category_id'] = df['class_' + level].factorize()[0]
    category_id_df = df[['class_' +
                        level, 'category_id']].drop_duplicates().sort_values(
                            'category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'class_' +
                          level]].values)

    # Train Features extraction
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2',
                            encoding='latin-1', ngram_range=(1, 2),
                            stop_words=None)
    features = tfidf.fit_transform(df.line).toarray()
    labels = df.category_id

    # Instantiate the machine learning classifiers
    if classifier == "svm":
        model = LinearSVC(dual=False)
    if classifier == "logit":
        model = LogisticRegression(max_iter=10000)
    if classifier == "gnb":
        model = GaussianNB()
    if classifier == "rf":
        model = RandomForestClassifier()
    if classifier == "dt":
        model = DecisionTreeClassifier()
    if classifier == "ada":
        model = AdaBoostClassifier()
    if classifier == "mlp":
        model = MLPClassifier(random_state=1, max_iter=300)
    if classifier == "xgb":
        model = XGBClassifier()
        # model = XGBClassifier(objective="multi:softprob", random_state=42)

    # Train model
    model.fit(features, labels)

    # save the model to disk

    pickle.dump(model, open(path_save_model, 'wb'))

    return(tfidf, model, id_to_category)


# Load ML model
def load_model(path_train, path_save_model, level, classifier):

    df = pd.read_csv(path_train, delimiter='\t')
    print("---------  Train class size  --------")
    if level == "level1":
        print(df.class_level1.value_counts())
    if level == "level2":
        print(df.class_level2.value_counts())
    if level == "level3":
        print(df.class_level3.value_counts())
    if level == "level4":
        print(df.class_level4.value_counts())
    if level == "level12":
        print(df.class_level12.value_counts())
    if level == "level123":
        print(df.class_level123.value_counts())
    if level == "level1234":
        print(df.class_level1234.value_counts())
    print("-------------------------------------")

    df['category_id'] = df['class_' + level].factorize()[0]
    category_id_df = df[['class_' +
                        level, 'category_id']].drop_duplicates().sort_values(
                            'category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'class_' + level]]
                          .values)
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=1, norm='l2',
                            encoding='latin-1', ngram_range=(1, 2),
                            stop_words=None)
    features = tfidf.fit_transform(df.line).toarray()
    model = pickle.load(open(path_save_model, 'rb'))

    return(tfidf, model, id_to_category)


# test ML
def test(path_test, tfidf, model, level):

    # test
    df2 = pd.read_csv(path_test, delimiter='\t')
    df2 = df2[pd.notnull(df2['class_' + level])]
    col2 = ['class_' + level, 'line']
    df2 = df2[col2]

    df2['category_id'] = df2['class_' + level].factorize()[0]
    category_id_df2 = df2[['class_' +
                          level, 'category_id']].drop_duplicates().sort_values(
                              'category_id')
    category_to_id2 = dict(category_id_df2.values)
    id_to_category2 = dict(category_id_df2[['category_id', 'class_' + level]]
                           .values)

    print("---------  Test class size  ---------")
    if level == "level1":
        print(df2.class_level1.value_counts())
    if level == "level2":
        print(df2.class_level2.value_counts())
    if level == "level3":
        print(df2.class_level3.value_counts())
    if level == "level4":
        print(df2.class_level4.value_counts())
    if level == "level12":
        print(df2.class_level12.value_counts())
    if level == "level123":
        print(df2.class_level123.value_counts())
    if level == "level1234":
        print(df2.class_level1234.value_counts())

    print("-------------------------------------")
    # Test Features extraction
    test = tfidf.transform(df2.line).toarray()
    labels_test = (df2.category_id)

    y_pred = model.predict(test)

    return(y_pred, labels_test, id_to_category2)


# Generate predictions
def generate_predictions(path_test, path_pred, tab_pred):

    with codecs.open(path_pred, 'w', encoding='utf-8') as fout:
        with codecs.open(path_test, 'r', encoding='utf-8') as f:
            cpt = 0
            ind = 0
            correct = 0
            # skip first line
            next(f)
            for line in f:

                ch = line.strip().split('\t')
                new_line = ch[1]
                pred = tab_pred[ind]
                ind += 1

                if ch[2].rstrip() == pred:
                    correct += 1

                ch = new_line + '\t' + pred + "\t" + ch[2].rstrip()
                fout.write(ch + '\n')
                # print(ch)
                cpt += 1

            acc = (correct / (cpt))*100
            # print("Accuracy = " + str(acc))
    return(str(acc))


# evaluation functions ML
def eval(y_pred, labels_test, id_to_category, id_to_category2):
    i = 0
    acc = 0
    for x in labels_test:

        if id_to_category2[x] == id_to_category[y_pred[i]]:
            acc += 1
        i += 1

    print("Accuracy = " + str((acc/i)*100))


# load reference or prediction text
def load_text(path):

    tab_ref = {}

    with codecs.open(path, 'r', encoding='utf-8') as f:
        cpt = 0
        pred = ""
        preds = []
        ok = 0
        nb1 = 0
        for line in f:
            seg = line.rstrip()
            if seg == "==========":
                ok = 1
                nb1 += 1
                cpt += 1
            else:
                if ok == 1:
                    if pred == "":
                        pred = "1"
                    else:
                        pred += "1"
                        preds.append(cpt)
                    ok = 0
                else:
                    preds.append(cpt)
                    if pred == "":
                        pred = "0"
                    else:
                        pred += "0"
        pred += "1"
    return pred, nb1, preds


# Segmentation functions
def load_ref_labels(path_train, level):

    tab_annotations = {}
    with codecs.open(path_train, 'r', encoding='utf-8') as f:
        # Flat
        # line_number	line	class_level1	class_level2	class_level3
        # Hierarchical
        # line_number	line	class_level1	class_level12	class_level123
        next(f)

        for line in f:
            # Get classes
            classes_ = line.strip().split('\t')
            if level == "level1":
                tab_annotations[classes_[2]] = classes_[2]
            if level == "level2" or level == "level12":
                tab_annotations[classes_[3]] = classes_[3]
            if level == "level3" or level == "level123":
                tab_annotations[classes_[4]] = classes_[4]

        labels_tmp = []
        for x in tab_annotations:
            labels_tmp.append((x))

    return(labels_tmp)


def segmentation(path_pred, path_seg_pred, labels, relaxation):
    tab_seg_ind = []
    tab_seg_label = {}
    document = {}

    # load test document
    with codecs.open(path_pred, 'r', encoding='utf-8') as f:
        ind = 1

        for line in f:
            document[ind] = line
            ind += 1

    max_sequence_begin = {}
    max_len_sequence = {}

    begin = 0
    end = 0
    for label in labels:

        # print ("Gold Label is " + label)
        ind = 1
        max_ = 0
        ind_max = 1
        max_sequence_begin = {}
        max_sequence_begin[label] = ind
        max_len_sequence[label] = 0

        predecessor = "none"
        all_sequence_begin = []
        for i in range(1, len(document)):

            line = document[i]

            ch = line.strip().split('\t')
            pred = ch[1]
            new_line = ch[0]

            if i > 1:
                predecessor = document[i-1].split('\t')[1]
            else:
                predecessor = "none"

            if pred == label:

                if predecessor == pred:  # keep incrementing
                    max_ += 1
                else:
                    # start a new count
                    ind_max = i
                    max_ = 1
            else:
                # save the sequence length if higher than the preceeding one
                if predecessor == label:

                    if max_ > 0:  # save current sequence

                        if max_ >= 2:  # max_len_sequence[label]:

                            if ind_max not in max_sequence_begin:

                                max_sequence_begin[ind_max] = max_
                                all_sequence_begin.append(ind_max)

            ind += 1

        # print ("max sequence begin :" + str(max_sequence_begin[label]))
        # print (all_sequence_begin)
        max_ = 0
        ind_max = 0
        cpt = 1
        for ind in all_sequence_begin:

            # print (str(ind) + " ____ "+ str(max_sequence_begin[ind]))
            if max_ < max_sequence_begin[ind]:
                max_ = max_sequence_begin[ind]
                ind_max = cpt

            cpt += 1

        ind_max = ind_max-1  # because sequence tab indice starts from 0
        if ind_max >= 0:
            # print ("maaax ind " + str(ind_max))
            # print ("maaax ind " + str(ind_max) + '\t'
            #        + str(all_sequence_begin[ind_max]))

            # extract sequence starting from the longest subseqence:
            ind_current = ind_max

            # go left for the beginning of the segment:
            i = ind_max
            if i == 0:
                begin = all_sequence_begin[i]
            else:

                for i in range(ind_max, 0, -1):
                    if i-1 >= 0:
                        error = abs(all_sequence_begin[i] -
                                    all_sequence_begin[i-1])
                        if error < relaxation:

                            begin = all_sequence_begin[i-1]
                        else:
                            begin = all_sequence_begin[i]

                            break

            # go right for the end of the segment:
            i = ind_max

            if i == len(all_sequence_begin):
                end = all_sequence_begin[i]
            else:

                for i in range(ind_max, len(all_sequence_begin)-1):

                    if i+1 <= len(all_sequence_begin)-1:
                        error = abs(all_sequence_begin[i] -
                                    all_sequence_begin[i+1])
                        if error < relaxation:
                            end = all_sequence_begin[i+1]
                        else:
                            end = all_sequence_begin[i]
                            break

            # print ("----> Begin : " + str(begin))
            tab_seg_ind.append(begin)
            tab_seg_label[begin] = label
            # print ("----> End : "   + str(end))

    tab = sorted(tab_seg_ind)
    for x in tab:
        print(str(x) + " " + tab_seg_label[x])

    with codecs.open(path_seg_pred, 'w', encoding='utf-8') as fout:
        fout.write("=========="+'\n')
        first = 0
        for i in range(1, len(document)+1):

            if i in tab_seg_label:
                if first > 0:
                    fout.write("=========="+'\n')
                else:
                    first += 1
            line = document[i].split('\t')[0]
            fout.write(line + '\n')


# eval_segmentation
def eval_segmentation(path_ref, path_pred):

    ref, nbref1, refs = load_text(path_ref)
    pred, nbpred1, preds = load_text(path_pred)

    d = {"stargazer": {"1": refs, "2": preds}}

    seg1 = d['stargazer']['1']
    seg2 = d['stargazer']['2']
    segs1 = se.convert_positions_to_masses(seg1)
    segs2 = se.convert_positions_to_masses(seg2)
    print("pk\tWindowdiff: \n")
    print(str(round(se.pk(segs2, segs1), 4)) + "\t" +
          str(round(se.window_diff(segs2, segs1), 4)))


# Generate line classification
def gen_line_classification(directory, path_train, bool_train, path_test, path_pred,
                            path_save_model, classifier, level, relaxation, data_type):
    # MAIN
    if bool_train:

        (tfidf, model, id_to_category) = train(path_train, path_save_model, level, classifier)
    else:
        # Load model ...
        print("Load saved model... ")
        (tfidf, model, id_to_category) = load_model(path_train, path_save_model, level, classifier)

    onlyfiles = [f for f in listdir(path_test) if isfile(
                    join(path_test, f))]

    for test_file in onlyfiles:
        print("=====================================")
        print(test_file)
        (y_pred, labels_test, id_to_category2) = test(path_test + test_file, tfidf, model, level)

        eval(y_pred, labels_test, id_to_category, id_to_category2)

        tab_pred = {}
        cpt = 0
        for i in range(len(labels_test)):

            tab_pred[cpt] = id_to_category[y_pred[i]]
            cpt += 1
        pred_file = "/" + test_file + "_" + level + ".pred_" + classifier
        generate_predictions(path_test + test_file, path_pred + pred_file, tab_pred)
        path_out_pred = directory + "/test/pred/" + test_file + "_" + level + ".pred_" + classifier
        test_noext = splitext(test_file)[0]
        path_seg_pred = directory + "/test/seg/" + test_noext + "_" + level + ".pred_" + classifier
        labels = load_ref_labels(path_train, level)
        segmentation(path_out_pred, path_seg_pred, labels, relaxation)
        ch = directory + "/test/choiformat/"
        path_ref = ch + data_type + "/" + test_noext + "_" + level + ".ref"
        eval_segmentation(path_ref, path_seg_pred)
