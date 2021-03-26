# Horae
This repository provides source codes for two applications: 
  - i)  Text detection and alignment such as psalms, orations, etc.
  - ii) Books of Hours segmentation.
# Requirements:
- python3.7
  see requirements.txt file
- pycodestyle 2.6.0 (pep8) (--max-line-length=100) 
  (ex: pycodestyle file.py --max-line-length=100)
- After installing **segeval**, open the file **windowdiff.py** (.local/lib/python3.7/site-packages/segeval/window/) and replace the following instruction:
       
       assert len(window) is window_size + 1 by assert len(window) == window_size + 1   
  
# Execution
- git clone https://github.com/hazemAmir/gitHorae.git
- cd Horae
- virtualenv -p python ENV
- source ENV/bin/activate 
- pip3 install -r requirements.txt

# Quick start
- python3 run_load_json.py
- python3 run_seg_preprocessing.py
- python3 run_segmentation.py -v True -r 100 -s False -t True -c svm -dt hier -l level1 

## Preprocessing
   1) ### Load and parse JSON volumes (python3 run_load_json.py)
       
       Input: 2 input directories (TEKLIA JSON format) 
      
             - Manual annotations (*.json) located in "data/horae-json-export/manual_annotations/volumes_name.json" 
             - Transcriptions (*.json) located in "data/horae-json-export/transcriptions/volumes_name*.json"         
  
       Output:  1 output directory:
       
              - Alignment directory (contains raw files used to detect and extract liturgical texts such as psalms, etc.)
                       --> location: ../data/alignment/raw/ conctains multicolumn csv files
                       --> raw files columns format: transcription '\t' image_id '\t' element_id '\t' element_polygon
              - 1 log file that contains information about the aligned volumes and the number of aligned sections
                       --> Location: ../data/alignment/alignment.log
              
   2) ### Prepare data files for segmentation (python3 run_seg_preprocessing.py)   
      
      Input: 2 input directories: choose train and test files from "../data/segmentation/raw/" and copy them into the following directories:  
              
              - Train directory: ../data/segmentation/train/raw/
              - Test directory:  ../data/segmentation/test/raw/
      
      Output: 7 output directories (contain several files format dedicated to train and test line classification and segmentation)
         
             --> location: ../data/segmentation/....
                           "../data/segmentation/train/csv/hier/"
                           "../data/segmentation/train/csv/flat/"
                           
                           "../data/segmentation/test/csv/hier/"
                           "../data/segmentation/test/csv/flat/"
                           "../data/segmentation/test/choiformat/hier/"
                           "../data/segmentation/test/choiformat/flat/"
                           "../data/segmentation/test/txt/"
## Segmentation
 1) python3 run_segmentation.py -v True -r 100 -s False -t True -c svm -dt hier -l level1
    performs training with svm, bert or bert2 and produces segmentaton results for all the volums contained in the ../data/segmentation/test/raw/ directory
   
   ## Parameters:
   
    - Segmentation Level -->  --level or -l (level1/level12/level123)
    - Segmentation type --> --data_type or -dt 'hierarchical or flat segmentation (hier/flat)
    - Classifier --> --classifier or -cl (svm, bert ,bert2)
    - Train --> --train or -t  (True/False)
    - Send --> --send or -s  'Send annotation to Arkindex' (True/False)
    - Validation--> --valid or -v (True/False)
    - Relaxaion factor -->  --relaxation or -r 'Relaxation factor berween 50 and 100'
