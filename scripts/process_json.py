# -*- coding: utf-8 -*-
#  Project: HORAE
#
#  Author: Amir HAZEM
#  Created: 09/09/2020
#  Updated: 25/03/2021
#  Role: Load JSON files library
#
# Libraries.
from __future__ import division
import codecs
import argparse
from os import listdir
from os.path import isfile, join
import sys
from collections import OrderedDict
import json
import cv2
from shapely.geometry import Polygon
import numpy as np
from operator import itemgetter, attrgetter


# Functions

# Converts a complex polygon to a 4 coordinates polygon
def convert_poly(element_polygon):
    rect = cv2.boundingRect(np.float32(element_polygon))
    x = rect[0]
    y = rect[1]
    w = rect[2]
    h = rect[3]
    x1 = int(x)
    y1 = int(y)
    x2 = int(x)
    y2 = int(y1 + h)
    x3 = int(x1 + w)
    y3 = int(y1 + h)
    x4 = int(x1 + w)
    y4 = int(y1)
    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    return(polygon)


# Converts a string polygon to a Polygon object
def str2polygon(poly_str):

    ch = str(poly_str).split(',')
    x1 = int(ch[0].split("[[")[1])
    y1 = int(ch[1].split("]")[0])
    x2 = int(ch[2].split("[")[1])
    y2 = int(ch[3].split("]")[0])
    x3 = int(ch[4].split("[")[1])
    y3 = int(ch[5].split("]")[0])
    x4 = int(ch[6].split("[")[1])
    y4 = int(ch[7].split("]")[0])

    polygon = Polygon([(x1, y1), (x2, y2), (x3, y3), (x4, y4)])
    return(polygon)


# converts a Polygon object to string
def polygon2str(element_polygon):
    cpt = 0
    ch = element_polygon.split(',')
    coord1 = ch[0].split("((")[1]
    x1 = int((coord1).split(' ')[0])
    y1 = int((coord1).split(' ')[1])
    coord2 = (ch[1].strip()).split(" ")
    x2 = int((coord2)[0])
    y2 = int((coord2)[1])
    coord3 = (ch[2].strip()).split(" ")
    x3 = int((coord3)[0])
    y3 = int((coord3)[1])
    coord4 = (ch[3].strip()).split(" ")
    x4 = int((coord4)[0])
    y4 = int((coord4)[1])
    poly_str = "[[" + str(x1) + ", " + str(y1) + "], [" + str(x2) + ", "\
        + str(y2) + "], [" + str(x3) + ", " + str(y3) + "], ["\
        + str(x4) + ", " + str(y4) + "], [" + str(x1) + ", "\
        + str(y1) + "]]"
    return(poly_str)


# Load annotated volumes
def load_annotator(path_annotator):

    tab_volume = {}
    nuplet_image_id = []

    onlyfiles = [f for f in listdir(path_annotator) if isfile(
        join(path_annotator, f))]
    cpt_section = 0
    for filename in onlyfiles:

        if filename.endswith('.json'):
            print(filename)
            with open(path_annotator + '/' + filename) as fjson:
                data = json.load(fjson, object_pairs_hook=OrderedDict)

                for p in data:
                    # Get volume info
                    if str(p) == "id":
                        volume_id = data[p]
                    if str(p) == "name":
                        volume_name = data[p]
                    if str(p) == "corpus":
                        corpus_id = data[p]['id']
                        corpus_name = data[p]['name']
                    if str(p) == "pages":
                        for k in data[p]:
                            if str(k['subelements']) != '[]':
                                page_id = k['id']
                                page_name = k['name']
                                image_id = k['image']['id']
                                image_url = k['image']['url']
                                for ll in k['subelements']:
                                    element_id = ll['id']
                                    element_type = ll['type']
                                    element_class = ll['annotation']
                                    element_polygon = ll['polygon']
                                    cpt_section += 1
                                    nuplet_image_id.append((volume_name,
                                                            image_id,
                                                            element_polygon,
                                                            element_class,
                                                            element_type,
                                                            page_id,
                                                            element_id))
                                tab_volume[volume_name] = volume_name
    print("The total number of sections is = " + str(cpt_section))
    return(tab_volume, nuplet_image_id)


# Load transcription volumes
def load_transcriptions(path_transcriptions):

    tab_volume = {}
    nuplet_image_id = []

    onlyfiles = [f for f in listdir(path_transcriptions) if isfile(
        join(path_transcriptions, f))]
    for filename in onlyfiles:

        if filename.endswith('.json'):
            print(filename)
            with open(path_transcriptions + '/' + filename) as fjson:
                data = json.load(fjson, object_pairs_hook=OrderedDict)
                for p in data:
                    # Get volume info
                    if str(p) == "id":
                        volume_id = data[p]
                    if str(p) == "name":
                        volume_name = data[p]
                        tab_volume[volume_name] = volume_name
                        print(volume_name)
                    if str(p) == "corpus":
                        corpus_id = data[p]['id']
                        corpus_name = data[p]['name']
                    if str(p) == "classifications":
                        for l1 in data[p]:
                            image_type = l1['class']

                    if str(p) == "pages":
                        for k in data[p]:
                            if str(k['transcriptions']) != '[]':
                                page_id = k['id']
                                page_name = k['name']
                                image_id = k['image']['id']
                                image_url = k['image']['url']

                                for l2 in k['transcriptions']:
                                    transcription_id = l2['id']
                                    transcription_text = l2['text']
                                    for l3 in l2['text_line']:
                                        element_id = l2['text_line']['id']
                                        element_polygon = l2['text_line']['polygon']
                                        element_polygon_rect = convert_poly(element_polygon)

                                    nuplet_image_id.append((volume_name,
                                                            image_id,
                                                            element_polygon_rect,
                                                            transcription_text,
                                                            page_id, image_type,
                                                            element_id))

    return(tab_volume, nuplet_image_id)


# Get volume's image
def get_img_by_volume(volume, tab_triplet_image_id_anno1):
    tab_image_id = []
    for x in tab_triplet_image_id_anno1:
        if x[0] == volume:
            image_id = x[1]
            element_polygon = x[2]
            element_class = x[3]
            page_id = x[4]
            element_id = x[6]
            tab_image_id.append((image_id, element_polygon, element_class, element_id, page_id))
    return(tab_image_id)


# Align sections with their corresponding transcriptions
def align_sections(path_boh_annotator, path_transcriptions, path_raw,
                   path_out, tab_volume_annot, tab_volume_trans,
                   tab_nuplet_annot, tab_nuplet_trans, path_log):
    nb_annot = 0
    nb_all = 0
    with codecs.open(path_log + '/alignment.log', 'w', encoding='utf-8') as flog:
        for volume in tab_volume_annot:
            flog.write("========" + '\n')
            nb_all += 1
            if volume in tab_volume_trans:
                print(volume)
                flog.write(volume + '\n')
                nb_annot += 1
                tab_annot_by_volume = get_img_by_volume(volume, tab_nuplet_annot)
                tab_trans_by_volume = get_img_by_volume(volume, tab_nuplet_trans)

                count = 0
                tab_section = []
                for x in tab_annot_by_volume:
                    img_id_x = x[0].rstrip()
                    polygon_x = str2polygon(x[1])
                    class_x = x[2]
                    element_id_x = x[3]
                    page_id_x = x[4]
                    surface = 0
                    ok = 0
                    for y in tab_trans_by_volume:

                        img_id_y = y[0].rstrip()

                        if str(img_id_x) == str(img_id_y):

                            polygon_y = y[1]
                            trans_y = y[2]
                            element_id_y = y[3]
                            page_id_y = y[4]
                            intersection = polygon_x.intersection(polygon_y)
                            if intersection.area > surface:
                                surface = intersection.area
                                save_polygone = y[1]
                                save_img_id_y = img_id_y
                                save_trans_y = trans_y
                                save_page_id_y = page_id_y
                                ok = 1

                    if ok == 1:
                        count += 1
                        class_ = x[2]
                        tab_section.append((save_img_id_y, save_polygone,
                                           class_, save_trans_y, save_page_id_y))
                    else:
                        s = "no match"
                        flog.write("Missed section\t" + class_x + '\n')
                        print("Missed section\t" + class_x)
                total = len(tab_annot_by_volume)
                print("Aligned sections/Total = " + str(count) + "/" + str(total))
                flog.write("Aligned sections/Total = " + str(count) + "/" + str(total) + '\n')
                if count > 0 and count == total:
                    tab_class_by_page_id = {}
                    for element in tab_section:
                        polygon = polygon2str(str((np.asarray(element[1]))))
                        class_ = element[2]
                        page_id = element[4]
                        key = page_id + '\t' + polygon
                        if key in tab_class_by_page_id:
                            tab_class_by_page_id[key] = tab_class_by_page_id[key] + '___' + class_
                        else:
                            tab_class_by_page_id[key] = class_

                    with codecs.open(path_out + '/' + volume, 'w', encoding='utf-8') as fout:
                        with open(path_raw + '/' + volume) as fraw:
                            for line in fraw:
                                ch = line.strip().split('\t')
                                key = ch[2] + '\t' + ch[3]
                                if key in tab_class_by_page_id:
                                    class_ = tab_class_by_page_id[key]
                                    fout.write(line.strip() + '\t' + class_ + '\n')
                                else:
                                    fout.write(line.strip() + '\n')
                else:
                    print("Section alignment failed for:\t" + volume)
                    flog.write("Section alignment failed for:\t" + volume + '\n')
            else:
                print("No corresponding json file transcription for: " + volume)
                flog.write("No corresponding json file transcription for: " + volume + '\n')
        flog.write("========" + '\n')
        print("Number of aligned volumes/total = " + str(nb_annot) + "/" + str(nb_all))
        flog.write("Number of aligned volumes/total = " + str(nb_annot) + "/" + str(nb_all) + '\n')


# get each section level
def get_sections(sections):
    ch = sections.split("___")
    tag1 = "| L1 |"
    tag2 = "| L2 |"
    tag3 = "| L3 |"
    tag4 = "| L4 |"
    tab_tag = {}
    tab_sec = {}
    tab_tag[tag1] = ""
    tab_tag[tag2] = ""
    tab_tag[tag3] = ""
    tab_tag[tag4] = ""
    sec1 = ""
    sec2 = ""
    sec3 = ""
    sec4 = ""
    for i in ch:
        for j in tab_tag:
            if j in i:
                tab_sec[j] = i

    for k in tab_sec:
        if tag1 in tab_sec[k]:
            sec1 = tab_sec[k]
        if tag2 in tab_sec[k]:
            sec2 = tab_sec[k]
        if tag3 in tab_sec[k]:
            sec3 = tab_sec[k]
        if tag4 in tab_sec[k]:
            sec4 = tab_sec[k]
    return(sec1, sec2, sec3, sec4)


# simplify annotations
def filter_sections(sections):
    ch = sections.split("___")
    filt_sec = ""
    for i in range(0, len(ch)-1):
        if i == 0:
            filt_sec = ch[i].split(' |')[0].strip()
        else:
            filt_sec = filt_sec + '\t' + ch[i].split(' |')[0].strip()
    return(filt_sec)


# sort sections and generates the corresponding files used for segmentation
def sort_sections(path_tagged_raw, volume, path_out_align, path_out_seg, tab_sec1,
                  tab_sec2, tab_sec3, tab_sec4, tab_sec12, tab_sec123):
    flag = 0
    nb_not_assigned = 0
    tmp_sec1 = ""
    tmp_sec2 = ""
    tmp_sec3 = ""
    tmp_sec4 = ""

    with codecs.open(path_out_align + '/' + volume, 'w', encoding='utf-8') as falign,\
         codecs.open(path_out_seg + '/' + volume, 'w', encoding='utf-8') as fseg:
        with open(path_tagged_raw + '/' + volume) as fraw:
            for line in fraw:
                ch = line.strip().split('\t')
                if len(ch) < 5 and flag == 0:
                    nb_not_assigned += 1
                else:
                    flag = 1
                    if len(ch) == 5:
                        (sec1, sec2, sec3, sec4) = get_sections(ch[4])

                        if sec1 != tmp_sec1 and sec1 != "":
                            tmp_sec1 = sec1
                            tmp_sec2 = ""
                            tmp_sec3 = ""
                            tmp_sec4 = ""

                        if sec2 != tmp_sec2 and sec2 != "":
                            tmp_sec2 = sec2
                            tmp_sec3 = ""
                            tmp_sec4 = ""

                        if sec3 != tmp_sec3 and sec3 != "":
                            tmp_sec3 = sec3
                            tmp_sec4 = ""

                        if sec4 != tmp_sec4 and sec4 != "":
                            tmp_sec4 = sec4

                    sections = tmp_sec1 + "___"

                    if tmp_sec2 != "":
                        sections += tmp_sec2 + "___"
                    if tmp_sec3 != "":
                        sections += tmp_sec3 + "___"
                    if tmp_sec4 != "":
                        sections += tmp_sec4

                    linealign = ch[0] + '\t' + ch[1] + '\t' + ch[2] + '\t' + ch[3] + '\t' + sections
                    filtered_sec = filter_sections(sections)

                    lineseg = ch[0] + '\t' + filtered_sec
                    falign.write(linealign + '\n')
                    fseg.write(lineseg + '\n')
                    # save sections
                    sec = filtered_sec.split('\t')

                    if len(sec) == 4:
                        if sec[0] in tab_sec1:
                            tab_sec1[sec[0]] += 1
                        else:
                            tab_sec1[sec[0]] = 1

                        if sec[1] in tab_sec2:
                            tab_sec2[sec[1]] += 1
                        else:
                            tab_sec2[sec[1]] = 1

                        if sec[2] in tab_sec3:
                            tab_sec3[sec[2]] += 1
                        else:
                            tab_sec3[sec[2]] = 1

                        if sec[3] in tab_sec4:
                            tab_sec4[sec[3]] += 1
                        else:
                            tab_sec4[sec[3]] = 1
                        tag_12 = sec[0] + "___" + sec[1]
                        if tag_12 in tab_sec12:
                            tab_sec12[tag_12] += 1
                        else:
                            tab_sec12[tag_12] = 1
                        tag_123 = sec[0] + "___" + sec[1] + "___" + sec[2]
                        if tag_123 in tab_sec123:
                            tab_sec123[tag_123] += 1
                        else:
                            tab_sec123[tag_123] = 1
                    else:
                        if len(sec) == 3:
                            if sec[0] in tab_sec1:
                                tab_sec1[sec[0]] += 1
                            else:
                                tab_sec1[sec[0]] = 1

                            if sec[1] in tab_sec2:
                                tab_sec2[sec[1]] += 1
                            else:
                                tab_sec2[sec[1]] = 1

                            if sec[2] in tab_sec3:
                                tab_sec3[sec[2]] += 1
                            else:
                                tab_sec3[sec[2]] = 1
                            tag_12 = sec[0] + "___" + sec[1]
                            if tag_12 in tab_sec12:
                                tab_sec12[tag_12] += 1
                            else:
                                tab_sec12[tag_12] = 1
                            tag_123 = sec[0] + "___" + sec[1] + "___" + sec[2]
                            if tag_123 in tab_sec123:
                                tab_sec123[tag_123] += 1
                            else:
                                tab_sec123[tag_123] = 1

                        else:
                            if len(sec) == 2:
                                if sec[0] in tab_sec1:
                                    tab_sec1[sec[0]] += 1
                                else:
                                    tab_sec1[sec[0]] = 1

                                if sec[1] in tab_sec2:
                                    tab_sec2[sec[1]] += 1
                                else:
                                    tab_sec2[sec[1]] = 1
                                tag_12 = sec[0] + "___" + sec[1]
                                if tag_12 in tab_sec12:
                                    tab_sec12[tag_12] += 1
                                else:
                                    tab_sec12[tag_12] = 1
                            else:
                                if len(sec) == 1:
                                    if sec[0] in tab_sec1:
                                        tab_sec1[sec[0]] += 1
                                    else:
                                        tab_sec1[sec[0]] = 1
    return(tab_sec1, tab_sec2, tab_sec3, tab_sec4, tab_sec12, tab_sec123)


# Save sections for BERT
def save_sections(path_class_seg, tab_sec1, tab_sec2, tab_sec3, tab_sec12, tab_sec123):
    class_file = "level1.txt"
    index = 0
    with codecs.open(path_class_seg + '/' + class_file, 'w', encoding='utf-8') as fout:
        for x in tab_sec1:
            fout.write(x + '\t' + str(index) + '\n')
            index += 1

    class_file = "level2.txt"
    index = 0
    with codecs.open(path_class_seg + '/' + class_file, 'w', encoding='utf-8') as fout:
        for x in tab_sec2:
            fout.write(x + '\t' + str(index) + '\n')
            index += 1

    class_file = "level3.txt"
    index = 0
    with codecs.open(path_class_seg + '/' + class_file, 'w', encoding='utf-8') as fout:
        for x in tab_sec3:
            fout.write(x + '\t' + str(index) + '\n')
            index += 1

    class_file = "level12.txt"
    index = 0
    with codecs.open(path_class_seg + '/' + class_file, 'w', encoding='utf-8') as fout:
        for x in tab_sec12:
            fout.write(x + '\t' + str(index) + '\n')
            index += 1

    class_file = "level123.txt"
    index = 0
    with codecs.open(path_class_seg + '/' + class_file, 'w', encoding='utf-8') as fout:
        for x in tab_sec123:
            fout.write(x + '\t' + str(index) + '\n')
            index += 1


# parse and sort a given volume
def extract_ordered_transcriptions(tab_img_id_by_volume, path_raw):

    current_volume_name = ""
    cpt = 0
    tab_image_id = []
    # print(current_volume_name)
    for x in tab_img_id_by_volume:
        if cpt == 0:
            tmp_volume_name = x[0]
            current_volume_name = x[0]
        else:
            tmp_volume_name = x[0]

        if tmp_volume_name == current_volume_name:
            tmp_image_id = x[1]
            tmp_element_polygon = x[2]
            tmp_element_class = x[3]
            tmp_element_id = x[4]
            tmp_image_type = x[5]
            tab_image_id.append((tmp_image_id, tmp_element_polygon,
                                tmp_element_class, tmp_element_id,
                                tmp_image_type))
        else:
            # sort the current volume
            sort_volume(path_raw, current_volume_name, tab_image_id)
            # next volume
            # initialize tab_image_id
            tab_image_id = []
            tmp_volume_name = x[0]
            current_volume_name = x[0]
            tmp_image_id = x[1]
            tmp_element_polygon = x[2]
            tmp_element_class = x[3]
            tmp_element_id = x[4]
            tmp_image_type = x[5]
            tab_image_id.append((tmp_image_id, tmp_element_polygon,
                                tmp_element_class, tmp_element_id,
                                tmp_image_type))
        cpt += 1
    # sort last volume
    sort_volume(path_raw, current_volume_name, tab_image_id)


# save a given page
def write_page(sorted_sent, current_image, current_element_id, current_image_id, fout):

    for x in sorted_sent:
        element_polygon = x[0]
        proba = x[1]
        ch = current_image[element_polygon]
        ch_element_id = current_element_id[ch]
        ch_image_id = current_image_id[ch]
        t = ch + '\t' + ch_image_id + '\t' + ch_element_id + '\t' + polygon2str(element_polygon)
        fout.write(t + '\n')


# gen min and max X1 axis
def get_x1_min_max(current_image):
    x_min = 111111110
    x_max = 0
    for line in current_image:
        ch = line.split(',')
        coord1 = ch[0].split("((")[1]
        x1 = int((coord1).split(' ')[0])

        if x1 < x_min:
            x_min = x1
        if x1 > x_max:
            x_max = x1
    return(x_min, x_max)


# split image into left and right pages
def get_left_right_image(current_image):
    tab_page_left = {}
    tab_page_right = {}
    tab_page_all = {}
    (x_min, x_max) = get_x1_min_max(current_image)
    mid = (x_max - x_min)
    for line in current_image:

        ch = line.split(',')
        coord1 = ch[0].split("((")[1]
        x1 = int((coord1).split(' ')[0])
        y1 = int((coord1).split(' ')[1])
        if x1 <= mid:  # same page
            tab_page_left[line] = int(y1)
        else:
            tab_page_right[line] = int(y1)
        # in case it is not a double page
        tab_page_all[line] = int(y1)
    return(tab_page_left, tab_page_right, tab_page_all)


# sort a given volume
def sort_page(current_image, current_element_id, current_image_id, image_type, fout):
    tab_page1 = {}
    tab_page2 = {}
    tab_page = {}
    tab_sort_left = []
    tab_sort_right = []
    tab_sort = []
    sorted_page_left = []
    sorted_page_right = []
    # put lines in the left or the right page before sorting
    (tab_page1, tab_page2, tab_page) = get_left_right_image(current_image)

    if image_type == "double_page":

        for a in tab_page1:
            tab_sort_left.append((a, tab_page1[a]))
        sorted_page_left = sorted(tab_sort_left, key=itemgetter(1), reverse=False)
        # even if a double page exists it doesn't mean that text is present
        if len(sorted_page_left) > 0:
            write_page(sorted_page_left, current_image,
                       current_element_id, current_image_id, fout)

        for a in tab_page2:
            tab_sort_right.append((a, tab_page2[a]))
        sorted_page_right = sorted(tab_sort_right, key=itemgetter(1), reverse=False)

        if len(sorted_page_right) > 0:
            write_page(sorted_page_right, current_image,
                       current_element_id, current_image_id, fout)
    else:

        for a in tab_page:
            tab_sort.append((a, tab_page[a]))

        sorted_page = sorted(tab_sort, key=itemgetter(1), reverse=False)
        write_page(sorted_page, current_image, current_element_id, current_image_id, fout)


# parse all volumes for sorting each image
def sort_volume(path_raw, current_volume_name, tab_annot_by_volume):
    cpt = 0
    current_image = {}
    current_element_id = {}
    current_image_id = {}
    sorted_page_left = []
    sorted_page_right = []
    inner_cpt_simple_page = 0
    inner_cpt_double_page = 0
    cpt_simple_page = 0
    cpt_double_page = 0
    with codecs.open(path_raw + '/' + current_volume_name, 'w', encoding='utf-8') as fout:

        for x in tab_annot_by_volume:
            if cpt == 0:
                previous_image_id = x[0]

            image_id = x[0]
            element_polygon = str((np.asarray(x[1])))  # polygon
            element_class = x[2]  # transcription
            element_id = x[3]
            image_type = x[4]
            cpt += 1

            if str(image_id) == str(previous_image_id):
                current_image[element_polygon] = element_class
                current_element_id[element_class] = element_id
                current_image_id[element_class] = image_id
            else:

                # All lines of the current page were gathered
                # Here we sort them according to polygon element
                # We take into account the double pages (which we consider as a double column
                # in a single page)
                sort_page(current_image, current_element_id, current_image_id, image_type, fout)
                # save next image
                current_image = {}
                current_element_id = {}
                current_image_id = {}
                previous_image_id = x[0]
                element_id = x[3]
                current_image[element_polygon] = element_class
                current_element_id[element_class] = element_id
                current_image_id[element_class] = previous_image_id

        # process the last page
        sort_page(current_image, current_element_id, current_image_id, image_type, fout)
