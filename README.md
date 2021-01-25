# Horae
This repository provides source codes for two applications: 
  - i)  Text detection and alignment such as psalms, orations, etc.
  - ii) Books of Hours segmentation.
# Requirements:
- python3.5
  see requirements.txt file
- pycodestyle 2.6.0 (pep8) (--max-line-length=100) 
  (ex: pycodestyle file.py --max-line-length=100)
 
# Execution
- git clone https://github.com/hazemAmir/gitHorae.git
- cd Horae
- virtualenv -p python ENV
- source ENV/bin/activate 
- pip3 install -r requirements.txt

# Quick start
- python3 run_load_json.py
- python3 run_seg_preprocessing.py
- python3 run_segmentation.py 

## Preprocessing
   1) python3 run_load_json.py
       
       Input: 2 input directories (TEKLIA JSON format) 
      
             - Manual annotations (*.json) located in "data/horae-json-export/manual_annotations/volumes_name.json" 
             - Transcriptions (*.json) located in "data/horae-json-export/transcriptions/volumes_name*.json"         
  
       Output:  1 output directory:
       
       - alignment directory (contains raw files that will be used to detect and extract liturgical texts such as psalms and other liturgical pieces...)
         --> location: ../data/alignment/raw/ conctains multicolumn csv files
         --> raw files columns contain: transcription '\t' image_id '\t' element_id '\t' element_polygon
                     
   2) python3 run_seg_preprocessing.py   
      
      Input: 2 input directories: Choose train and test files from "../data/alignment/raw/" and copy them into the floowing directories  
              - Train directory: ../data/segmentation/train/raw/
              - Test directory ../data/segmentation/test/raw/
              
      - segmentation directoy (contains several files format dedicated to train and test the lines classification and segmentation)
         --> location: ../data/segmentation/
                     
             "../data/segmentation/train/csv/hier/"
             "../data/segmentation/train/csv/flat/"
             "../data/segmentation/test/csv/hier/"
             "../data/segmentation/test/csv/flat/"
             "../data/segmentation/test/choiformat/hier/"
             "../data/segmentation/test/choiformat/flat/"
             "../data/segmentation/test/txt/"
          

