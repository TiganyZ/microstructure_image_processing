#!/usr/bin/python3

""" Script which combines the functions of image_processing and image_analysis

Processing and data extraction pipeline: 

  process_images.py -> analyse_images.py -> preprocess_data.py -> classification.py 
(image modification) (raw data extraction) (data analysis/cleaning) (Model training and prediction)

"""

from process_images import *
from analyse_images import *
from preprocess_data import *
from classification import *

import copy
import sys

plot=False

# Get arguments from commandline
if len(sys.argv) > 1:
    # We have arguments
    if len(sys.argv) > 2:
        print("ERROR: Wrong number of arguments given.\n Please give directory of images")
        exit(1)
    else:
        image_directory = sys.argv[1]

else:
    image_directory = "images"

original_image_directory = copy.deepcopy(image_directory)

##############################
###--- Start processing ---###

print(f"Starting to process images from {image_directory}...")

# processes = [RemoveBakelite, WhiteBackgroundRemoval,  HistogramEquilization]

processes = [ RemoveBakeliteBoundary, WhiteBackgroundRemoval, OtsuThreshold]
arguments = [{} for process in processes ] 


for process, args in zip(processes, arguments):
    pc = ProcessContainer(process, args, image_directory)

    print(f"> Processing using {pc.method_name} method")
    pc.process_images(plot=plot)
    
    image_directory = pc.dir_name


# image_directory = "images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"

##############################
###--- Start Extracting ---###

print(f"Analysing images from {image_directory}...")    
# Now analyse
print(f"> Alpha-beta fraction")    
analysis = AlphaBetaFraction
ac = AnalysisContainer(analysis, image_directory, original_image_directory)
ac.analyse_images(plot=plot)
data_directory = ac.dir_name



#################################
###--- Start Preprocessing ---###

print(f"Starting to preprocess data from {data_directory}...")
denudation_data = "denudation_data.dat"
print(f"Preprocessing data from {data_directory}, with images from {original_image_directory}..")
# Now analyse
print(f"> Gradient data from alpha-beta fraction")
analysis =  MultisectionFit #BisectionFit
ppc = PreprocessContainer(analysis, original_image_directory, data_directory, denudation_data)
ppc.analyse_images(plot=plot)
preprocess_directory = ppc.dir_name
data = ppc.path



##################################
###--- Start Classification ---###
print(f"Starting to classify data from {data}...")
excel_data = 'excel_data.dat'
print(f"Analysing images from {original_image_directory}, using data {data}\n >with excel data {excel_data}..")
trainers = [TrainLogisticRegression, TrainSVM]
cc = ClassificationContainer(trainers, data, original_image_directory, excel_data, use_excel=1)
cc.train_models(plot=plot)



# print(f"> Chris Alpha-beta fraction")    
# analysis = ChrisAlphaBetaFraction
# ac = AnalysisContainer(analysis, image_directory, original_image_directory)
# ac.analyse_images(plot=plot)


