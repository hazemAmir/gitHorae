# -*- coding: utf-8 -*-
from __future__ import division
import codecs
import argparse
import nltk
import ast
import sys
import re
from collections import OrderedDict
from operator import itemgetter, attrgetter
from arkindex import ArkindexClient
import logging
from apistar.exceptions import ErrorResponse
from scipy import spatial
import numpy as np
from operator import add
import argparse

logging.basicConfig(
    format='[%(levelname)s] %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
cli = ArkindexClient('469804ae46dcce16155efd5fe9cb08bcca3eceb0')

# Functions:
# load_args()
# send_class(...)

# Load arguments.


def load_args():
    parser = argparse.ArgumentParser(description='Get parameters...')
    parser.add_argument('--th', '-t', help='threshold to select a\
    segment as a candidate', action="store", dest="th", default="40")
    parser.add_argument('--perc_overlap', '-p', help='Segment overlap in\
    percentage', action="store", dest="perc_overlap", default="25")
    parser.add_argument('--overlap', '-o', help='Segment \
    overlap', action="store", dest="overlap", default="True")
    args = parser.parse_args()
    return(args)


def send_class_depricated(target, class_label, confidence_score, page_image,
               corpus_id, transcription_polygon):

    # page_image,page_id,corpus_id,transcription_polygon
    class_name = class_label
    confidence = round(confidence_score/100, 4)
    # from the classes list of IRHT
    classifier = "ls2n"
    # FROM THE JSON EXPORT
    page_image = page_image  # 'af4fc50c-df6c-4287-a0de-4fd67b99f68a'
    page_id = target  # 'bc61c7eb-a8b7-4f89-a2d5-9a8488c51255'
    corpus_id = corpus_id  # '731bf02d-f45e-4af7-af43-23ae3145048d'
    trans_polygon = transcription_polygon

    # CREATE A TEXT LINE ELEMENT FROM THE TRANSCRIPTION COORDINATES
    body = {

                "type": "text_segment",
                "name": class_name,
                "image": page_image,
                "corpus": corpus_id,
                "parent": page_id,
                "polygon": trans_polygon,
                "source": classifier
            }

    target_id = ""
    try:
        new_line = cli.request('CreateElement', body=body, slim_output=True)
        target_id = new_line['id']
        logger.info('Text line element {} created on page {}.'
                    .format(target_id, page_id))
    except ErrorResponse as e:
        logger.error(f"{e.status_code}, {e.title}, {e.content}")

    # TARGET : THE CREATED ELEMENT UUID FROM ARKINDEX
    target = target_id
    # CLASSES = list of dicts
    classes = [
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    # only True for manually created classifications
                    "high_confidence": False,
                }
            ]

    # BODY : PARENT, SOURCE and CLASSES

    body = {
            "parent": target,
            "classifier": classifier,
            "classifications": classes,
           }

    try:
        sent = cli.request('CreateClassifications', body=body)
        logger.info('Classification {} sent to element {}.'
                    .format(class_name, target))
    except ErrorResponse as e:
        logger.error(f"{e.status_code}, {e.title}, {e.content}")


def send_class(target, class_label, element_type, confidence_score, page_image,
               corpus_id, transcription_polygon):

    # page_image,page_id,corpus_id,transcription_polygon
    class_name = class_label
    confidence = round(confidence_score/100, 4)
    # from the classes list of IRHT
    classifier_id = "a014510a-cbee-4b30-8a7b-8e3004556579"
    # FROM THE JSON EXPORT
    page_image = page_image  # 'af4fc50c-df6c-4287-a0de-4fd67b99f68a'
    page_id = target  # 'bc61c7eb-a8b7-4f89-a2d5-9a8488c51255'
    corpus_id = corpus_id  # '731bf02d-f45e-4af7-af43-23ae3145048d'
    trans_polygon = transcription_polygon

    # CREATE A TEXT LINE ELEMENT FROM THE TRANSCRIPTION COORDINATES
    body = {

                "type": element_type,
                "name": class_name,
                "image": page_image,
                "corpus": corpus_id,
                "parent": page_id,
                "polygon": trans_polygon,
                "worker_version": classifier_id
            }

    target_id = ""
    try:
        new_line = cli.request('CreateElement', body=body, slim_output=True)
        target_id = new_line['id']
        logger.info('{} element {} created on page {}.'
                    .format(element_type, target_id, page_id))
    except ErrorResponse as e:
        logger.error(f"{e.status_code}, {e.title}, {e.content}")

    # TARGET : THE CREATED ELEMENT UUID FROM ARKINDEX
    target = target_id
    # CLASSES = list of dicts
    classes = [
                {
                    "class_name": class_name,
                    "confidence": confidence,
                    "high_confidence": False
                }
            ]

    # BODY : PARENT, SOURCE and CLASSES

    body = {
            "parent": target,
            "classifier": classifier_id,
            "classifications": classes
           }

    try:
        sent = cli.request('CreateClassifications', body=body)
        logger.info('Classification {} sent to element {}.'
                    .format(class_name, target))
    except ErrorResponse as e:
        logger.error(f"{e.status_code}, {e.title}, {e.content}")


# Clean text lines:
# - Tokenizes
# - lowercases
# - removes the punctuation
def clean(text):

    line_ = (text.strip())
    sent_tok = (' '.join(nltk.word_tokenize(line_.lower())))
    text_line = ' '.join(x for x in sent_tok.split(' ') if x.isalpha())

    return(text_line.strip())


def load_texts(path, tab_text):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        # with codecs.open(path, 'r', encoding='latin-1') as f:
        # skip the head line
        tab_text = {}
        tab_labels = {}
        next(f)
        for line in f:
            # print(line)
            string_ = line.strip().split('\t')  # input text line
            id_text = string_[0]  # id text
            tab_text[id_text] = clean(string_[1])  # text
            tab_labels[id_text] = string_[3]  # text label
    return(tab_text, tab_labels)

# Load reference list (Gold standard annotations)
def load_ref_list(path,tab_ref):
    
    with  codecs.open(path,'r',encoding = 'utf-8') as f:
        # skip the head line
        next(f)
        for line in f:
            string_ = line.split('\t')          

            volume   = string_[0]
            psalm    = string_[1]
            url_ps   = string_[4].rstrip()
            input_ = volume +'\t' +  psalm + '\t'+ url_ps
            tab_ref[input_] =  input_ 

    return tab_ref

# Load volume
def load_volume(path_volume, tab_text_line, tab_image_id, tab_page_id,
                tab_polygone):

    with codecs.open(path_volume, 'r', encoding='utf-8') as f:
        i = 0
        for line in f:
            cline = line.strip()
            tab_text_line[i] = clean(cline.split('\t')[0])  # text_line
            tab_image_id[i] = cline.split('\t')[1]  # image_id
            tab_page_id[i] = cline.split('\t')[2]  # page_id
            # polygone_element
            tab_polygone[i] = ast.literal_eval(((cline.split('\t')[3].rstrip())
                                                ))
            i += 1
    return(tab_text_line, tab_image_id, tab_page_id, tab_polygone)


# Split a volume into blocs:
def bloc_split_old(tab_text_line, len_bloc, tab_blocs, tab_blocs_begin, tab_page_id):

    current_bloc = ""
    index = 0
    i = 0
    count_page_step = 0  # allows a n page step for blocs gathering
    for i in range(0, len(tab_text_line)):
        
        if i == 0:
            num_line = 0
            current_page_id = tab_page_id[i].strip() 
        
        text_line = tab_text_line[i].strip()  # remove spaces
        tmp_page_id = tab_page_id[i].strip()
        
        if current_page_id != tmp_page_id:
            count_page_step += 1
        
        #print(count_page_step)

        ch = text_line.split(' ')
        
        # due to punctuation filtering some times empty lines appear
        if len(ch) > 1:# or (len(ch) == 1 and ch[0] != ''):
            
            # particular case if text_line > size bloc
            if len(ch) >= len_bloc:
                print("eeeeeeee")
                sys.exit()
                tab_blocs[index] = text_line
                tab_blocs_begin[index] = num_line
                # save index bloc the begining
                num_line = i  # the beginning of next bloc
                count_page_step = 0
                current_bloc = ""
                index += 1
            else:    
                if len(ch) + len(current_bloc.split(' ')) <= len_bloc and count_page_step < 3: # 2 step pages  

                    current_bloc = current_bloc + ' ' + text_line

                else:
                    print("-----> " + str(count_page_step) + " --- " + str(len(current_bloc.split(' '))) + " --- " + str(len_bloc))
                    print(current_bloc)
                    # save_bloc
                    tab_blocs[index] = current_bloc
                    tab_blocs_begin[index] = num_line
                    # save index bloc the begining
                    num_line =  i  # the beginning of next bloc
                    current_bloc = text_line
                    current_page_id = tmp_page_id
                    count_page_step = 0
                    index += 1

    # save last bloc:
    tab_blocs[index] = current_bloc
    tab_blocs_begin[index] = num_line

    return(tab_blocs, tab_blocs_begin)


# Split a volume into blocs:
def bloc_split(tab_text_line, len_bloc, tab_blocs, tab_blocs_begin, tab_page_id):

    current_bloc = ""
    index = 0
    i = 0
    count_page_step = 0  # allows a n page step for blocs gathering
    for i in range(0, len(tab_text_line)):
        j = 0
        stop = 0
        #print("----> " + str(i))
        while i + j  < len(tab_text_line) and stop == 0:

            if i == 0:
                num_line = 0
                current_page_id = tab_page_id[i + j].strip() 
        
            text_line = tab_text_line[i + j].strip()  # remove spaces
            tmp_page_id = tab_page_id[i + j].strip()
        
            if current_page_id != tmp_page_id:
                count_page_step += 1
        
            #print(count_page_step)

            ch = text_line.split(' ')
            #print(j)
            # due to punctuation filtering some times empty lines appear
            if len(ch) > 1 or (len(ch) == 1 and ch[0] != ''):
              
                # particular case if text_line > size bloc
                if len(ch) >= len_bloc:
                    print("eeeeeeee")
                    sys.exit()
                    tab_blocs[index] = text_line
                    tab_blocs_begin[index] = num_line
                    # save index bloc the begining
                    num_line = i + 1  # the beginning of next bloc
                    count_page_step = 0
                    current_bloc = ""
                    index += 1
                else:    
                    if len(ch) + len(current_bloc.split(' ')) <= len_bloc: # and count_page_step < 3: # 2 step pages  

                        current_bloc = current_bloc + ' ' + text_line
 
                    else:
                        #print("-----> " + str(count_page_step) + " --- " + str(len(current_bloc.split(' '))) + " --- " + str(len_bloc))
                        #print(current_bloc)
                        # save_bloc
                        tab_blocs[index] = current_bloc
                        tab_blocs_begin[index] = num_line
                        # save index bloc the begining
                        num_line =  i + 1 # the beginning of next bloc
                        current_bloc = text_line
                        current_page_id = tmp_page_id
                        count_page_step = 0
                        index += 1
                        stop = 1

            j += 1
        
    # save last bloc:
    
            if i + j  > len(tab_text_line):
                print("----> " + str(i))
            #k = 0
            #text_line = tab_text_line[i].strip()

            #while i + k < len(tab_text_line)   
            #tab_blocs[index] = current_bloc
            #tab_blocs_begin[index] = num_line

    return(tab_blocs, tab_blocs_begin)

# Split a volume into blocs with overlap:
def bloc_split_with_overlap(txt, tab_text_line, len_bloc, tab_blocs,
                            tab_blocs_begin, perc_overlap):
    current_bloc = ""
    index = 0
    i = 0
    not_found = 0
    save_ind = 0
    perc_ = perc_overlap  # 25% 50% 75%
    # print("bloc split with overlap")
    # print(len(tab_text_line))
    while i < len(tab_text_line):
        #print(i)
        #print(len(tab_text_line))
        if i == 0:
            num_line = 0

        text_line = tab_text_line[i].strip()  # remove spaces

        ch = text_line.split(' ')
        #print('------')
        #print(text_line)
        #print(len(ch))
        #print(len_bloc)
        #print(len(ch) + len(current_bloc.split(' ')))
        #print("-"+current_bloc+"-")
        #print(i)
        #print(txt)
        
        if len(ch) > 1 or  (len(ch) == 1 and ch[0] != ''):

            if int(len(ch) + len(current_bloc.split(' '))) <= len_bloc:

                current_bloc = current_bloc + ' ' + text_line
                current_bloc = current_bloc.strip()
                # save index n% of the bloc
                #print("eeeee")
                #print(current_bloc)
                
                if (len(ch) + len(current_bloc.split(' '))) >=\
                round(perc_ * (len_bloc/100)) and not_found == 0:
                    save_ind =  i # i+1
                    not_found = 1
                    #print("eeeee")
                
                i += 1

            else:



                #print("else")
                #print(i)

                if current_bloc == "":
                    #print("ooooooooooooo")
                    current_bloc = text_line
                    #num_line = i
                    not_found = 1
                    save_ind =  i 
                
                else:
                    if not_found == 0:
                    	# bloc == line 
                        #num_line = i
                        save_ind =  i 
                    
                         
                #print("am here")
                
                #print(current_bloc)
                # print(len(current_bloc))
                #print(ch)
                # save_bloc
                tab_blocs[index] = current_bloc
                tab_blocs_begin[index] = num_line
                # save index bloc the begining
                num_line = save_ind  # i # the beginning of next bloc
                i = save_ind + 1  # to avoid loop when text = 1 line
                current_bloc = ""
                not_found = 0
                index += 1

        else:
            i += 1

    # save last bloc:
    tab_blocs[index] = current_bloc
    tab_blocs_begin[index] = num_line
    return(tab_blocs, tab_blocs_begin)


# Return the vocabulary of a given text bloc
def get_vocab(input_text, tab_vocab):

    ch = input_text.split(' ')

    for i in ch:
        tab_vocab[i] = i

    return(tab_vocab)


# Intersect blocs and references in terms of vocabulary
def intersect_blocs(tab_vocab_bloc_i, tab_vocab_ref):
    cpt = 0

    for x in tab_vocab_bloc_i:

        if x in tab_vocab_ref:
            cpt += 1
    score = (cpt / len(tab_vocab_ref))*100

    return score


def extract_bloc(tab_text_line, index, len_bloc):

    text_bloc = ""
    last_ind = 0
    stop = 0
    for j in range(index, index + len_bloc):

        if j < len(tab_text_line):
            text_line = tab_text_line[j].split(' ')

            if len(text_line) + len(text_bloc.split(' ')) <= len_bloc:

                text_bloc = text_bloc + ' ' + tab_text_line[j]

    return(text_bloc)


def rescale_blocs(tab_text_line, score, current_bloc, len_bloc,
                  index, tab_vocab_ref):

    # Get Backward
    stop = 0
    found = 0
    i = index - 1
    new_score = score

    while i >= 0 and stop == 0:

        new_bloc = extract_bloc(tab_text_line, i, len_bloc)
        tab_vocab_new = {}
        tab_vocab_new = get_vocab(new_bloc, tab_vocab_new)
        tmp_new_score = intersect_blocs(tab_vocab_new, tab_vocab_ref)

        if tmp_new_score >= new_score:
            new_bloc_begin = i
            found = 1
            new_score = tmp_new_score
        else:
            stop = 1
        i = i - 1
        current_bloc = new_bloc

    # Get Forward
    stop = 0
    i = index + 1

    while i < len(tab_text_line) and stop == 0:

        new_bloc = extract_bloc(tab_text_line, i, len_bloc)
        tab_vocab_new = {}
        tab_vocab_new = get_vocab(new_bloc, tab_vocab_new)
        tmp_new_score = intersect_blocs(tab_vocab_new, tab_vocab_ref)

        if tmp_new_score >= new_score:
            new_bloc_begin = i
            found = 1
            new_score = tmp_new_score
        else:
            stop = 1
        i = i + 1
        current_bloc = new_bloc

    if found == 1:
        return(new_bloc_begin, new_score)
    else:
        return(index, score)

def compute_P_R_F1(tab_annot_links,tab_ref):
    acc = 0
    F1  = 0     
    
    #for k in tab_ref:    
    #    print(k)
    #print("-----------------")    
    #for i in tab_annot_links:
    #    print(i)
    #    print("-----------------")
    #    sys.exit()

    for i in tab_annot_links:
        #print(tab_annot_links[i])
        #print(i)
        # remove score
        ch = tab_annot_links[i].split('\t') 
        j = ch[0] + '\t' + ch[1] + '\t'+ ch[2]
        #j = ch[1] + '\t'+ ch[2]
        #print(tab_ref)
        #print(j)
        if j in tab_ref:
            acc+=1
            #sys.exit()
    P  = (acc/len(tab_annot_links))*100
    R  = (acc/(len(tab_ref)))*100
    if P+R > 0:
        F1 = 2 * P * R / (P+R)
    else:
        F1 = 0

    #print ("Precision = " + str(round(P,2)) )   
    #print ("Recall = " + str(round(R,2)))
    #print ("F1-measure = " + str( round(F1,2))  )   
    #print("Match ")
    #print(acc)
    #print("len Precision tab_annot_links")
    #print(len(tab_annot_links))
    #print("Len Recall tab ref")
    #print(len(tab_ref))
    print(str(round(P,2))+" & " +str(round(R,2)) + " & " + str( round(F1,2)))