#!/usr/bin/python3

""" Script which combines the functions of image_processing and image_analysis"""

from process_images import *
from analyse_images import *
import sys

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

    
print(f"Starting to process images from {image_directory}..")


processes = [RemoveBakelite, WhiteBackgroundRemoval,  OtsuThreshold]

for process in processes:
    pc = ProcessContainer(process, image_directory)

    print("> Processing using {pc.method_name} method")
    pc.process_images(plot=True)
    
    image_directory = pc.dir_name


print(f"Analysing images from {image_directory}..")    
# Now analyse
analysis = AlphaBetaFraction
ac = AnalysisContainer(analysis, image_directory)
ac.analyse_images(plot=True)
    

