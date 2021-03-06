#!/usr/bin/python3

"""Script which combines the functions of image_processing, image
analysis and image classification for the purposes of classifying beta
denudation.


The processing and data extraction pipeline is as follows: 

  process_images.py -> analyse_images.py -> preprocess_data.py -> classification.py 
(image modification) (raw data extraction) (data analysis/cleaning) (Model training and prediction)

By default, if an option is chosen to run, the dependent steps will run as well. 

Use the flag --only to run only the specified steps in the command line.


###--- Processing ---###

To process images, one can run the python script as 

`python3 beta_den.py --process`
or 
`python3 beta_den.py -p`

Images for processing are, by default, in a subdirectory of the
current folder called 'images'.

The name of the image directory can be specified by adding the command
line argument to the program -i <directory-name>.  

For example if I have images in a directory named FanBladeImages, one can process the
images by running

`python3 beta_den.py -p -i FanBladeImages`

The image processing will create directories of images which are
processed, these are for reference so one can see what is going on,
furthermore they help with repeatability.

"""

from process_images import *
from analyse_images import *
from preprocess_data import *
from classification import *

import argparse
import copy
import sys



# Create the parser
parser = argparse.ArgumentParser( description='Software for beta denudation analysis')

# Add the arguments
parser.add_argument('-i', '--image_directory',
                    type=str, 
                    help='Directory of the image directory for processing/analysing', default="images")


parser.add_argument('-p', '--process',
                    action='store_true',
                    help='Enables processing of images in image directory' )



parser.add_argument('-a', '--analyse',
                    action='store_true',
                    help='Enables analysis of volume fraction of processed images. \nIf process is a command line argument, the analysis directory is that which the end of processing produces. If process is not, the images analysed are from default image directory or --image-directory.' )


parser.add_argument('-d', '--data',
                    type=str, 
                    help='Directory/file of the data for extraction/classification', default="data.txt")


parser.add_argument('-e', '--extract',
                    action='store_true',
                    help='Enables extraction of beta volume fractions from --data_directory. If --analyse is a command line argument, then directory analysed is that which is analysis produces.' )


parser.add_argument('-f', '--classification_file',
                    type=str, 
                    help='File which contains list of image names and classifications in the first and second colums respectively', default="classification.txt")


parser.add_argument( '-c', '--classify',
                    action='store_true',
                    help='Train and use machine learning models to classify an images based on a surface to bulk ratio an image. Needs data file -d/--data to be passed in and for excel data to be found in the directory. If not passed in, or if extract argument is passed, the data will be the file which results from extraction' )


parser.add_argument( '-s', '--show',
                    action='store_true',
                    help='Plot all results from each of the steps (process/analysis/extract/classify).' )


parser.add_argument('-o', '--only',
                    action='store_true',
                    help='Enables only those command line arguments specified to run and not any dependent steps. If the arguments are --process --only then only processing occurs and none of the children tasks are run. ' )



# Execute parse_args()
args = parser.parse_args()


image_directory = args.image_directory
data_directory = args.data
data_file = args.data
only = args.only
classification_file = args.classification_file


original_image_directory = copy.deepcopy(image_directory)


plot=args.show


if only:
    process  = args.process
    analyse  = args.analyse
    extract  = args.extract
    classify = args.classify
else:
    process  = args.process
    analyse  = process or args.analyse
    extract  = analyse or args.extract
    classify = extract or args.classify


print(f"\n Starting beta_den\n > Images in directory '{image_directory}'\n > Data directory/file = '{data_directory}'\n > Classification file = '{classification_file}'\n > Showing visuals of steps? {plot} \n > Processing? {process} \n > Analysing? {analyse} \n > Extracting? {extract} \n > Classifying? {classify}\n ")
    
# Now check what steps we need to do based on the arguments

if not os.path.exists(image_directory):
    print(f"Error, image directory {image_directory} does not exist or has invalid name. \n Please put images in a directory named 'images' or pass in a directory of images with 'python3 {parser.prog} -i <image_directory> {-p / -a / -e / -s / -o}'")
    exit(1)
    

##############################
###--- Start processing ---###
if process:
    print(f"Starting to process images from {image_directory}...")

    processes = [ RemoveBakeliteBoundary, WhiteBackgroundRemoval, HistogramEquilization, OtsuThreshold]
    arguments = [{} for process in processes ] 


    for process, args in zip(processes, arguments):
        pc = ProcessContainer(process, args, image_directory)

        print(f"> Processing using {pc.method_name} method")
        pc.process_images(plot=plot)

        image_directory = pc.dir_name


##############################
###--- Start Extracting ---###


if analyse:
    print(f"Analysing images from {image_directory}...")    
    # Now analyse
    print(f"> Alpha-beta fraction")    
    analysis = AlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory, original_image_directory)
    ac.analyse_images(plot=plot)
    data_directory = ac.dir_name



#################################
###--- Start Preprocessing ---###
if extract:
    if not os.path.exists(data_directory):
        print(f"Error, data directory {data_directory} does not exist or has invalid name. \n Please pass in a directory of data with 'python3 {parser.prog} -e -d <data_directory>' or use that which analyse produces. ")
        exit(1)

    print(f"Starting to preprocess data from {data_directory}...")
    print(f"Preprocessing data from {data_directory}, with images from {original_image_directory}..")
    # Now analyse
    print(f"> Surface-to-bulk ratio from alpha-beta fraction")
    analysis =  MultisectionFit #BisectionFit
    ppc = PreprocessContainer(analysis, original_image_directory, data_directory)
    ppc.analyse_images(plot=plot)
    preprocess_directory = ppc.dir_name
    data = ppc.path



##################################
###--- Start Classification ---###
if classify:
    if not os.path.exists(data):
        print(f"Error, data file {data} does not exist or has invalid name. \n Please pass in a file of data with 'python3 {parser.prog} -c -d <data_file>' or use that which extract produces. ")
        exit(1)

    if not os.path.exists(classification_file):
        print(f"Error, classification data file {classification_file} does not exist or has invalid name. \n Please pass in a file of data with 'python3 {parser.prog} -i <image_directory> -f <classification_data_file>' or use that which extract produces. This file must have the images in the first column, surface-to-bulk ratio in the second column, and classification (0 for normal and 1 for denudated) in the last column ")
        exit(1)

    print(f"\nStarting to classify data from '{data}'...")
    print(f"\n Classifying images from {original_image_directory}, using data {data}\n > with classification data '{classification_file}'...")
    trainers = [TrainLogisticRegression, TrainSVM]
    cc = ClassificationContainer(trainers, data, original_image_directory, classification_file)
    cc.train_models(plot=plot)



# print(f"> Chris Alpha-beta fraction")    
# analysis = ChrisAlphaBetaFraction
# ac = AnalysisContainer(analysis, image_directory, original_image_directory)
# ac.analyse_images(plot=plot)


