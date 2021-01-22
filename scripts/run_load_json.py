# -*- coding: utf-8 -*-
#  Project: HORAE
#
#  Author: Amir HAZEM
#  Created: 09/09/2020
#  Updated: 22/01/2021
#  Role: Load JSON files
#

# Libraries.
from __future__ import division
import codecs
from os import listdir
from os.path import isfile, join
import sys
import process_json as pj


if __name__ == '__main__':

    # Inputs:
    path_annotator = "../data/horae-json-export/manual_annotations/"
    path_transcriptions = "../data/horae-json-export/transcriptions/"

    # Outputs
    # for alignment
    path_raw = "../data/alignment/raw/"
    path_tagged_raw = "../data/alignment/tagged_raw/"
    path_full_tagged_raw = "../data/alignment/full_tagged_raw/"
    path_log = "../data/alignment/"
    # for segmentation
    path_raw_seg = "../data/segmentation/raw/"

    try:
        print("Transcription Extraction Processing... ")

        (tab_volume_trans,
            tab_nuplet_trans) = pj.load_transcriptions(path_transcriptions)

        pj.extract_ordered_transcriptions(tab_nuplet_trans, path_raw)
    except Exception as exp:
        print("Transcription Extraction Error ", sys.exc_info()[0])
        print(str(exp))

    try:
        print("Section Alignment Processing...")
        (tab_volume_annot,
            tab_nuplet_annot) = pj.load_annotator(path_annotator)
        pj.align_sections(path_annotator, path_transcriptions, path_raw,
                          path_tagged_raw, tab_volume_annot, tab_volume_trans,
                          tab_nuplet_annot, tab_nuplet_trans, path_log)

    except Exception as exp:
        print("Alignment Unexpected error ", sys.exc_info()[0])
        print(str(exp))

    try:

        print("Line Sections Assignment...")
        onlyfiles = [f for f in listdir(path_tagged_raw) if isfile(
                    join(path_tagged_raw, f))]

        for volume in onlyfiles:
            pj.sort_sections(path_tagged_raw, volume, path_full_tagged_raw, path_raw_seg)

    except Exception as exp:
        print("Alignment Unexpected error ", sys.exc_info()[0])
        print(str(exp))
