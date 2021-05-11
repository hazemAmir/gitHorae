# -*- coding: utf-8 -*-
#  Project: HORAE
#
#  Author: Amir HAZEM
#  Created: 09/09/2020
#  Updated: 29/04/2021
#  Role: Psalms and passage detection and Alignment.
#


# Libraries.
from __future__ import division
import codecs
import argparse
import horae_text_alignement as ho
from os import listdir
from os.path import isfile, join
import sys
from collections import OrderedDict
from operator import itemgetter, attrgetter
from arkindex import ArkindexClient
import logging
from apistar.exceptions import ErrorResponse
import json

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
cli = ArkindexClient('469804ae46dcce16155efd5fe9cb08bcca3eceb0')


def extract_texts(path_texts, path_labels, path_ref_list, url_init, send, validation):
    # Parameters
    tab_texts = {}
    tab_ref = {}
    tab_labels = {}
    tab_annot_links = {}
    tab_annot_links_rescal = {}
    # Load data sets

    # Load texts and labels
    tab_texts, tab_labels = ho.load_texts(path_texts, tab_texts)
    # print(tab_labels)

    # Load psalm's/texts reference list
    tab_ref = ho.load_ref_list(path_ref_list, tab_ref)

    # Parse Volumes
    onlyfiles = [f for f in listdir(path_boh) if isfile(join(path_boh, f))]

    head = "num_line\tVolume\tPsalm\tURL\tConfidence Score\n"

    with codecs.open(path_pred, 'w', encoding='utf-8') as fout:
        fout.write(head)
        for filename in onlyfiles:

            tab_annot_psaume_number = []
            tab_annot_line_number = []
            print(filename)

            path_volume = path_boh + filename
            # tmp parameters
            tab_text_line = {}
            tab_image_id = {}
            tab_page_id = {}
            tab_polygone = {}
            tab_target = {}
            tab_class_label = {}
            tab_confidence_score = {}
            tab_page_image = {}
            tab_transcription_polygon = {}

            # Load Volume
            (tab_text_line, tab_image_id, tab_page_id,
             tab_polygone) = ho.load_volume(path_volume, tab_text_line,
                                            tab_image_id, tab_page_id,
                                            tab_polygone)

            cptt = 0
            for txt in tab_texts:  # Parse each text
                cptt += 1

                tab_blocs = {}
                tab_blocs_begin = {}

                text = tab_texts[txt]
                # split into block each reference texts's size in nb words
                len_bloc = len(text.split(' '))
                (tab_blocs, tab_blocs_begin) = ho.bloc_split(tab_text_line, len_bloc,
                                                             tab_blocs, tab_blocs_begin)

                # Load texts's vocabulary
                tab_vocab_text = {}
                tab_vocab_text = ho.get_vocab(text, tab_vocab_text)

                # contains the results of all the blocs for a given psalm
                tab_res_bloc_intersect = []
                for i in tab_blocs:  # Parse each bloc

                    # load bloc's vocabulary
                    tab_vocab_bloc_i = {}
                    tab_vocab_bloc_i = ho.get_vocab(tab_blocs[i],
                                                    tab_vocab_bloc_i)
                    # Get score
                    score = ho.intersect_blocs(tab_vocab_bloc_i,
                                               tab_vocab_text)
                    tab_res_bloc_intersect.append((i, score))

                # Sort the blocs

                sorted_intersect = sorted(tab_res_bloc_intersect,
                                          key=itemgetter(1), reverse=True)

                # Filter the results
                tab_bloc_selected = {}
                tab_bloc_selected_rescal = {}
                for x in sorted_intersect:

                    if x[1] >= th_intersect:

                        bloc_i = x[0]
                        score_intersect = round(x[1], 2)
                        algorithm = "word_overlap"
                        class_label = tab_labels[txt].rstrip()
                        target = tab_page_id[tab_blocs_begin[bloc_i]].rstrip()
                        page_image = tab_image_id[tab_blocs_begin
                                                  [bloc_i]].rstrip()
                        transcription_polygon = tab_polygone[tab_blocs_begin
                                                             [bloc_i]]
                        confidence_score = score_intersect

                        url = url_init + str(target)

                        res = filename + '\t' + str(txt) + '\t' + url

                        if int(bloc_i+1) not in tab_bloc_selected and int(bloc_i-1)\
                        not in tab_bloc_selected and int(bloc_i-2) not in tab_bloc_selected\
                        and int(bloc_i-2) not in tab_bloc_selected:

                            tab_annot_links[res] = res
                            tab_bloc_selected[bloc_i] = score_intersect

                # Rescale:
                tab_bloc_selected_new = {}
                for b in tab_bloc_selected:
                    # Get line begin
                    index_current_bloc = tab_blocs_begin[b]
                    current_bloc = tab_blocs[b]
                    score = tab_bloc_selected[b]

                    target = tab_page_id[index_current_bloc].rstrip()
                    transcription_polygon = tab_polygone[index_current_bloc]
                    url = url_init + str(target)

                    new_bloc, new_score = ho.rescale_blocs(tab_text_line,
                                                           score, current_bloc,
                                                           len_bloc,
                                                           index_current_bloc,
                                                           tab_vocab_text)

                    target = tab_page_id[new_bloc].rstrip()
                    transcription_polygon = tab_polygone[new_bloc]
                    url = url_init + str(target)

                    res = filename + '\t' + str(txt) + '\t' + url
                    res = res + '\t' + str(round(new_score, 2)) + "%"
                    bloc_i = new_bloc
                    ok = "yes"  # to alleviate multiple previous and
                    # next blocs selection which have a score > threshold
                    sizetext = len_bloc
                    for r in tab_bloc_selected_new:

                        proximity = abs(r-bloc_i)
                        if proximity <= sizetext:
                            ok = "no"

                    if ok != "no":
                        tab_bloc_selected_new[bloc_i] = bloc_i
                        tab_annot_links_rescal[res] = res

                        tab_annot_psaume_number.append((int(txt), res))
                        tab_annot_line_number.append((int(bloc_i), txt, res))
                        confidence_score = new_score

                        input_ = str(bloc_i) + " " + str(txt) + " " + res

                        tab_target[input_] = target
                        tab_class_label[input_] = class_label
                        tab_confidence_score[input_] = confidence_score
                        # algorithm
                        tab_page_image[input_] = page_image
                        # corpus_id
                        tab_transcription_polygon[input_] = transcription_polygon

            # Check duplicates before annotation
            check_duplicates = {}
            sorted_ = sorted(tab_annot_line_number,
                             key=itemgetter(0), reverse=False)
            for i in sorted_:
                bloc = i[0]
                num_text = i[1]
                link = i[2].split('\t')[2]
                score = i[2].split('\t')[3]
                link_all = i[2]
                input_ = str(bloc) + " " + str(num_text) + " " + link_all

                output = str(num_text) + '\t' + link_all

                if link not in check_duplicates:
                    fout.write(str(bloc) + '\t' + link_all+'\n')

                    print(num_text + '\t' + link + '\t' + str(score))
                    if send:
                        print("send")
                        ho.send_class(tab_target[input_], tab_class_label[input_],
                                      tab_confidence_score[input_],
                                      tab_page_image[input_], corpus_id,
                                      tab_transcription_polygon[input_])
                    check_duplicates[link] = link

        if validation:
            print("P R F1 scores Before rescaling")
            ho.compute_P_R_F1(tab_annot_links, tab_ref)
            # Rescale reslts
            # eval one volume
            print("P R F1 scores After rescaling")
            ho.compute_P_R_F1(tab_annot_links_rescal, tab_ref)


if __name__ == '__main__':

    #  Inputs:
    args = ho.load_args()
    #  Threshold to select a segment as a candidate  [40 - 95] %.
    th_intersect = float(args.th)
    validation = eval(args.valid)  # True / False (only if test annotations are available)
    send = eval(args.send)  # True / False (send annotations)

    # url arkindex
    url_init = "https://arkindex.teklia.com/element/"
    # Corpus id.
    corpus_id = '6dcf7706-f63d-4ede-9ffc-55b4f35faf2d'
    #  Raw manuscripts.
    path_boh = "../data/alignment/raw/"
    #  Reference texts psalms or other texts.

    if validation:
        path_texts = "../data/alignment/reference/textes_8psaumes_utf8.txt"
        path_boh = "../data/alignment/raw_test/"
    else:
        path_texts = "../data/alignment/reference/textes_liturgiques_utf8.txt"

    #  Reference list (human annotation) to compute accuracy.
    path_ref_list = "../data/alignment/reference/ref_penitential_psalms_8_boh.csv"
    #  Class annotations for psalms.
    path_labels = "../data/alignment/reference/table_etique_Psaumes.csv"

    # Outputs:
    path_pred = "./outputs/annotations_" + str(th_intersect) + ".csv"

    # Run extract text
    try:

        extract_texts(path_texts, path_labels, path_ref_list, url_init, send, validation)

    except Exception as exp:
        print("Unexpected Error ", sys.exc_info()[0])
        print(str(exp))
