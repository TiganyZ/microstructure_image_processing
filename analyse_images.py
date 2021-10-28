#!/usr/bin/python3.6

""" This file gets the alpha beta fraction from a portion of the file. 
"""

import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage import  exposure, img_as_float, img_as_uint, img_as_ubyte, util

from scipy import ndimage
from datetime import datetime

from abc import ABC, abstractmethod
import os
from typing import Type, TypeVar




# Make an abstract base class which supplies a processing function upon an image
class ImageAnalysis(ABC):
    
    @abstractmethod
    def analyse(self, image):
        pass
            
    @abstractmethod
    def plot(self, image):
        pass


class AlphaBetaFraction(ImageAnalysis):
    def __init__(self):
        self.name = "AlphaBetaFraction"
        self.comment = "#  pixel_depth  alphabetafraction   >  Analysis of alpha beta fraction"

    def analyse(self, image, sample_rate = 10, offset = 100):
        # Take in an an image file, which has been thresholded and has
        # the bakelite removed and then sample the alpha-beta volume
        # fraction

        n_samples = int((len(image) - offset)/float(sample_rate))
        
        self.data = np.zeros((n_samples, 2))
        index = 0
        for i, row in enumerate(image):
            if i > offset:
                if i != 0:
                    if i % sample_rate == 0:
                        # Get a row of pixels and find the mean for the
                        # alpha-beta volume fraction, where values closer
                        # to 1 are more alpha
                        self.data[index, 0] = i
                        self.data[index, 1] = np.mean(row)                    
                        index += 1
        
    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].plot(self.data[:,0], self.data[:,1])
        ax[1].set_xlabel("Surface Depth")
        ax[1].set_ylabel("Alpha-beta fraction")        
        ax[1].set_title('Alpha-beta volume fraction with surface depth')
        fig.tight_layout()

        plt.show()


        
        
        
# Make a containing class which takes a process image instance and
# then makes a corresponding directory for it

class AnalysisContainer:
    def __init__(self, analysis_method: Type[ImageAnalysis], image_directory: str):
        self.image_directory = image_directory
        self.images = os.listdir(self.image_directory)

        self.method = analysis_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")


    def analyse_images(self, plot=True):
        for image in self.images:
            original_image =  imageio.imread(f"{self.image_directory}/{image}", as_gray=True)
            self.method.analyse( original_image )
            if plot:
                self.method.plot(original_image)
            
            self.save(image, self.method.data)


    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.image_directory}_{self.dt}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, data):
        self.dir_name = self.create_output_directory()
        name = os.path.splitext(image_name)[0]
        path = os.path.join(self.dir_name, f"{self.method_name}_{name}.dat")

        print(f"> Saving data to {path}")
        with open(path, 'w') as f:
            f.write(self.method.comment + "\n")

            for row in self.method.data:
                data_line = (' '.join(["{: 12.6f}" * len(row)])).format( *tuple(row) ) + "\n"
                f.write(data_line)
        
            

        


if __name__ == '__main__':
    # Test out the functionality

    image_directory = "OtsuThreshold_WhiteBackgroundRemoval_RemoveBakelite_images"

    
    analysis = AlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory)
    ac.analyse_images(plot=True)
    
    
        
