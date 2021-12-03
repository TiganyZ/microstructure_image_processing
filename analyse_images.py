#!/usr/bin/python3.6

""" This file gets the alpha beta fraction from a portion of the file. 
"""

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
import imageio
import numpy as np
import re, copy
from skimage import color, exposure, img_as_float, img_as_uint, img_as_ubyte, util, measure
from scipy.optimize import curve_fit, minimize

from scipy import ndimage
from datetime import datetime

from abc import ABC, abstractmethod
import os
from typing import Type, TypeVar

from utils import calculate_boundary


# Make an abstract base class which supplies a processing function upon an image
class ImageAnalysis(ABC):
    
    @abstractmethod
    def analyse(self, image):
        pass
            
    @abstractmethod
    def plot(self, image):
        pass

class ChrisAlphaBetaFraction(ImageAnalysis):
    def __init__(self):
        self.name = "ChrisAlphaBetaFraction"
        self.comment = "#  pixel_depth  alphabetafraction   >  Analysis of alpha beta fraction"

    def get_fraction_from_line_segment(self, image, depth, coordinates):

        n_segments = len(coordinates)-1
        segment_fraction = np.zeros(n_segments)
        
        d = 0
        
        for i, coord in enumerate(coordinates):
            if i+1 == n_segments:
                break
            x1, y1 = coordinates[i  ,0], coordinates[i  ,1] + depth
            x2, y2 = coordinates[i+1,0], coordinates[i+1,1] + depth

            d = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

            # Should be a row, column format. Therefore, we have y then x
            line = measure.profile_line(image, (y1, x1), (y2, x2))

            segment_fraction[i] = line.sum() / d

        return np.mean(segment_fraction)
    
        
        
    def analyse(self, image, unprocessed_image,  sample_rate = 10, offset = 0):
        self.image = image
        # Get the actual proportion of bakelite from the top, doing the same analysis as chris.
        thresholded_image = unprocessed_image > 0.5*np.max(unprocessed_image)
        boundary, mean, sigma = calculate_boundary(thresholded_image,
                                                   sample_rate=40, offset = 0., n_sigma=0.5)
        self.lower_boundary = boundary 
        # boundary, mean, sigma = calculate_boundary(unprocessed_image>0.7, sample_rate = 40)

        self.image = image
        fixed_offset = 20
        offset = mean + fixed_offset + 0.5*sigma # + 2*sigma # np.max(lower_boundary[:,1])

        # Now find the difference in pixel heights between the images so we have true calibration
        removed_pixel_rows = unprocessed_image.shape[0] - image.shape[0]

        max_boundary = max(offset, removed_pixel_rows)

        offset = int(max_boundary) - removed_pixel_rows
        if offset < 0:
            print("A/B fraction analysis: WARNING: offset is less than zero, setting to 0. ")
            print("image.shape, unprocessed_image.shape, offset, mean, sigma, removed_pixel_rows, Y1, Y2")
            print(image.shape, unprocessed_image.shape, offset, mean, sigma, removed_pixel_rows, Y1, Y2)

            offset = 0

        # n_samples = int((len(image) - int(offset))/float(sample_rate))


        self.offset = offset

        self.surface_distances = []
        self.bulk_distances = []        
        Y1 = boundary[ 0,1]
        Y2 = boundary[-1,1]
        deltaY = Y2 - Y1

        Y1 = offset
        Y2 = offset + deltaY
        if Y2 < 0:
            Y1 -= deltaY
            Y2 -= deltaY

        self.data = ""
        

        Nsurf=10;
        Nbulk=22;
        surfpixels=3;
        bulkpixels=5;
        surfdist=Nsurf*surfpixels;
        bulkdist=Nbulk*bulkpixels;

        Y3=int(0.7*len(image))

        self.data += ('Surface_Ratios\n');

        SUM = np.zeros(Nsurf)
        Z=0
        idx=0
        while Z < surfdist:
            # Use the lower boundary to calculate
            print(image.shape, Y1, Y1+Z, Y2, Y2+Z)
            line = measure.profile_line(image, (Y1+Z,0), (Y2+Z, image.shape[1]-1)) # image[Z + Y3]
            SUM[idx] = line.sum() / 255. / len(line)
            #            SUM[idx] = self.get_fraction_from_line_segment(image, Z, boundary) / 255.
            self.surface_distances.append( (Y1+Y2)/2 + Z)
            print("SUM ", SUM[idx])
            self.data += (str(SUM[idx]) + "\n")
            Z += surfpixels
            idx += 1

        S_SUM_TOT=np.mean(SUM)
        S_SUM_SD=np.std(SUM)


        YY=1;
        B_SUM=np.zeros(Nbulk)
        self.data += ('Bulk_Ratios\n')

        Z=0
        idx=0
        while Z < bulkdist:
            line = measure.profile_line(image, (Y3-Z,0), ( Y3 + deltaY - Z, image.shape[1]-1 )) # image[Z + Y3]
            self.bulk_distances.append(Y3-Z)
            B_SUM[idx] =  line.sum() / 255. / len(line)
            print("B_SUM ", B_SUM[idx])
            self.data += (str(B_SUM[idx]) + "\n")
            Z += bulkpixels
            idx += 1

        B_SUM_TOT=np.mean(B_SUM)
        B_SUM_SD=np.std(B_SUM)


        FRatio=(S_SUM_TOT)/(B_SUM_TOT)

        self.data += ("#Average_surface"+"\n");
        self.data += (str(S_SUM_TOT)+"\n");
        self.data += ("#SD_surface"+"\n");
        self.data += (str(S_SUM_SD)+"\n");
        self.data += ("#Average_bulk")+"\n";
        self.data += (str(B_SUM_TOT)+"\n");
        self.data += ("#SD_bulk"+"\n");
        self.data += (str(B_SUM_SD)+"\n");
        self.data += ("#----------Final_Ratio----------\n");
        self.data += (str(FRatio)+"\n");
        self.data += ("#-------------------------------\n");

            

                        
    def plot(self, original_image):

        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
        ax[0].plot(self.lower_boundary[:,0], self.lower_boundary[:,1], 'r--', label = "Detected surface" )
        # ax[0].hlines(self.surface_distances, 0, 1, transform=ax[0].get_yaxis_transform(), label = "Sampled surface distances") 
        # ax[0].hlines(self.bulk_distances, 0, 1, transform=ax[0].get_yaxis_transform(), label = "Sampled bulk distances")       
        ax[0].legend(bbox_to_anchor=(0.15, -0.4, 0.6, -0.4), loc='lower left',
                      ncol=1, borderaxespad=0.)
        # ax[1].imshow(self.thresholded_image, cmap='gray')
        # ax[1].set_title('Thresholded image')

        
        ax[1].imshow(self.image, cmap='gray')
        ax[1].set_title('Processed image')
        ax[1].hlines(self.surface_distances, 0, 1, 'b', transform=ax[1].get_yaxis_transform(), label = "Sampled surface distances")
        ax[1].hlines(self.bulk_distances, 0, 1, 'm', transform=ax[1].get_yaxis_transform(), label = "Sampled bulk distances")
        ax[1].legend(bbox_to_anchor=(0.15, -0.4, 0.6, -0.4), loc='lower left',
                     ncol=1, borderaxespad=0.)

        
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        

        
        # y_fit, label = self.exp_analysis()
        # ax[1].scatter(self.data[:,0], self.data[:,1], label="data" )
        # ax[1].plot(self.data[:,0], y_fit, label=label)
        # ax[1].set_xlabel("Surface Depth")
        # ax[1].set_ylabel("Alpha-beta fraction")        
        # ax[1].set_title('Alpha-beta volume fraction with surface depth')
        # ax[1].legend()
        fig.tight_layout()
        plt.show()

    
class AlphaBetaFraction(ImageAnalysis):
    def __init__(self):
        self.name = "AlphaBetaFraction"
        self.comment = "#  pixel_depth  alphabetafraction   >  Analysis of alpha beta fraction"

    def get_fraction_from_line_segment(self, image, depth, coordinates):

        n_segments = len(coordinates)-1
        segment_fraction = np.zeros(n_segments)

        new_coordinates = copy.deepcopy(coordinates)
        new_coordinates[:,1] += 0.5*np.std(coordinates[:,1])
        
        d = 0
        
        for i, coord in enumerate(new_coordinates):
            if i+1 == n_segments:
                break
            x1, y1 = new_coordinates[i  ,0], np.int( new_coordinates[i  ,1] + depth) 
            x2, y2 = new_coordinates[i+1,0], np.int( new_coordinates[i+1,1] + depth) 

            d = np.sqrt( (x1-x2)**2 + (y1-y2)**2 )

            # Should be a row, column format. Therefore, we have y then x
            line = measure.profile_line(image, (y1, x1), (y2, x2))

            segment_fraction[i] = line.sum() / d

        return np.mean(segment_fraction)
            
    def analyse(self, image, unprocessed_image,  sample_rate = 1, offset = 0):
        # Take in an an image file, which has been thresholded and has
        # the bakelite removed and then sample the alpha-beta volume
        # fraction
        # p2, p98 = np.percentile(image, (2, 98))
        # rescaled_image = exposure.rescale_intensity(unprocessed_image, in_range=(p2, p98))
        self.thresholded_image = unprocessed_image > 0.5*np.max(unprocessed_image)
        self.lower_boundary, self.mean, self.sigma = calculate_boundary(self.thresholded_image,
                                                                        sample_rate=60, offset = 5, n_sigma=0)
        image = util.invert(image)
        self.image = image
        Y1 = self.lower_boundary[0,1]
        Y2 = self.lower_boundary[-1,1]
        deltaY = Y2-Y1

        self.n_sigma = 0
        offset = np.max(self.lower_boundary[:,1]) 
        # offset = self.mean #+ fixed_offset + 0.5*self.sigma # + 2*sigma # np.max(lower_boundary[:,1])

        # Now find the difference in pixel heights between the images so we have true calibration
        removed_pixel_rows = len(unprocessed_image) - len(image)

        max_boundary = max(offset, removed_pixel_rows)

        offset = int(max_boundary) - removed_pixel_rows

        # offset = int(offset) - removed_pixel_rows
        if offset < 0:
            print("A/B fraction analysis: WARNING: offset is less than zero, setting to 0. ")
            offset = 0
        
        n_samples = int((len(image) - int(offset))/float(sample_rate)) 

        
        self.offset = offset
        index = 0
        straight = False
        follow_boundary = not straight

        datax = []
        datay = []
        for i, row in enumerate(image):
            if i > offset:
                ni = i - int(offset) 
                if ni % sample_rate == 0:
                    if index > 20:
                        straight = True
                        follow_boundary = not straight


                    # alpha-beta volume fraction, where values closer
                    # to 1 are more alpha
                    datax.append( i )
                    if straight:
                        #                        datay.append( np.mean(image[i-int(sample_rate/2)-1:i+1])/255. )
                        datay.append( np.mean(image[i])/255. )
                    elif follow_boundary:
                        datay.append( self.get_fraction_from_line_segment(image, ni, self.lower_boundary) / 255. )
                    else:
                        if deltaY < 0:
                            i1 = i - int(deltaY)
                            i2 = i
                        else:
                            i1 = i
                            i2 = i + int(deltaY)
                        print(image.shape, i1, i2)

                        line = measure.profile_line(image, (i1,0), ( i2, image.shape[1]-1 ))
                        datay.append( line.sum() / 255. / len(line) )
                    index += 1

        self.data = np.zeros((index, 2))
        self.data[:,0] = np.asarray(datax)
        self.data[:,1] = np.asarray(datay)        


    def exp_decay(self, x, A, b):
        return A*np.exp(-b*x)

    def straight_line(self, x, m, c):
        return m*x + c

    
    def exp_analysis(self):
        # Given the data, and that we expect there to be a decay of
        # the proportion of alpha from the surface, the function one
        # can fit it os the exponential decay
        x, y = self.data[:,0], self.data[:,1]
        
        expected =  (1.0, 0.01)
        
        params,cov=curve_fit(self.exp_decay, x, y, expected)
        # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        stdevs = np.sqrt(np.diag(cov))# Calculate the residuals
        res = y - self.exp_decay(x, *params)

        label = f" {params[0]:3.1f}*exp(-{params[1]:3.1f}x) "
        
        return self.exp_decay(x, *params), label
        
    def linear_analysis(self):
        # Given the data, and that we expect there to be a decay of
        # the proportion of alpha from the surface, the function one
        # can fit it os the exponential decay
        x, y = self.data[:,0], self.data[:,1]
        
        expected =  (-1/2500., 0.7)
        
        params,cov=curve_fit(self.straight_line, x, y, expected)
        # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        stdevs = np.sqrt(np.diag(cov))# Calculate the residuals
        label = f" {params[0]:3.1e}*x + {params[1]:3.1e} "
        
        return self.straight_line(x, *params), label
        

                        
    def plot(self, original_image):
        
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), sharey=True)
        ax[0].plot(self.lower_boundary[:,0], self.lower_boundary[:,1], 'r', linestyle='dashed', linewidth=2, label = "Detected surface" )
        ax[0].hlines(self.offset, 0, 1, transform=ax[0].get_yaxis_transform(), linewidth=3,  label = f"Surf. μ + {self.n_sigma}σ")
        ax[0].legend(bbox_to_anchor=(0.15, -0.4, 0.6, -0.4), loc='lower left',
                      ncol=1, borderaxespad=0.)
        # ax[1].imshow(self.thresholded_image, cmap='gray')
        # ax[1].set_title('Thresholded image')
        
        ax[1].imshow(util.invert(self.image), cmap='gray')
        ax[1].plot(self.lower_boundary[:,0], self.lower_boundary[:,1], 'r', linestyle='dashed', linewidth=2, label = "Detected surface" )
        ax[1].set_title('Processed image')
        ax[1].hlines(self.offset, 0, 1, transform=ax[1].get_yaxis_transform(), linewidth=3,  label = f"Surf. μ + {self.n_sigma}σ")
        ax[1].legend(bbox_to_anchor=(0.25, -0.4, 0.7, -0.4), loc='lower left',
                     ncol=1, borderaxespad=0.)

        
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')


        # fig, ax = plt.subplots(nrows=1, ncols=2)

        # ax[0].imshow(original_image, cmap='gray')
        # ax[0].set_title('Original image')
        # ax[0].axis('off')

        
        try:
            y_fit, label = self.linear_analysis() # self.exp_analysis()
        except ValueError:
            print("Straight line fit: WARNING: Cannot fit to straight line, will give flat dependence for line")
            y_fit = [0.5 for i in self.data[:,0]]
            label = f" {0:3.1g}*x + {0.5:3.1g} "
        ax[2].plot(self.data[:,1], self.data[:,0], label="data" )
        ax[2].plot(y_fit, self.data[:,0], 'm-', label=label)
        # ax[2].set_ylabel("Surface Depth")
        ax[2].set_xlabel("Alpha-beta fraction")
        ax[2].set_title('α-β volume frac. vs depth')

        # asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
        # asp /= np.abs(np.diff(ax[1].get_xlim())[0] / np.diff(ax[1].get_ylim())[0])
        # ax[2].set_aspect(asp)

        # asp = np.diff(ax[2].get_xlim())[0] / np.diff(ax[2].get_ylim())[0]
        # ax[2].set_aspect(asp)
        ax[2].legend()
        ax[2].legend(bbox_to_anchor=(0.15, -0.4, 0.6, -0.4), loc='lower left',
                     ncol=1, borderaxespad=0.)
        plt.subplots_adjust(wspace=0, hspace=0)
        # ax[2].legend(bbox_to_anchor=(0.15, -0.4, 0.6, -0.4), loc='lower left',
        #              ncol=1, borderaxespad=0.)
        fig.tight_layout()
        plt.show()



class PrimaryAlphaSize(ImageAnalysis):
    def __init__(self):
        self.name = "PrimaryAlphaSize"
        comment_str = ' '.join( re.findall("[A-Z][^A-Z]*", self.name) )
        self.comment = "#  pixel_depth  {self.name}[pixels]   >  Analysis of {comment_str}"

    def analyse(self, image, unprocessed_image,  sample_rate = 10, offset = 100):
        # Analyse the image file from below the offset
        # > To find the primary alpha size, one can 

        
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
                        self.data[index, 1] = np.mean(image[i-int(sample_rate/4)-1:i+1,:])/255.                    
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
    def __init__(self, analysis_method: Type[ImageAnalysis], image_directory: str, original_image_directory: str):
        self.image_directory = image_directory
        self.original_image_directory = original_image_directory        
        self.images = sorted(os.listdir(self.image_directory))
        self.method = analysis_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")

    def get_filename_from_id(self, image):
        base_name = os.path.splitext(image.split('_')[0])[0]
        extension = os.path.splitext(image)[1]
        prefixed = [filename for filename in sorted(os.listdir(self.original_image_directory)) if filename.startswith(base_name)]

        # Assuming just one identifier from image name, can't deal with multiple images'
        print("Unprocessed image: ", prefixed[0])
        return prefixed[0]

    def analyse_images(self, plot=True):
        for image in self.images:
            original_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{image}"))
            unprocessed_name = self.get_filename_from_id(image)
            unprocessed_image =   color.rgb2gray(imageio.imread(f"{self.original_image_directory}/{unprocessed_name}"))
            self.method.analyse( original_image, unprocessed_image )
            if plot:
                self.method.plot(unprocessed_image)
            
            self.save(image, self.method.data)


    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.dt}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, data):
        self.dir_name = self.create_output_directory()
        name = os.path.splitext(image_name)[0]
        path = os.path.join(self.dir_name, f"{self.method_name}_{name}.dat")
        
        
        print(f"> Saving data to {path}")
        with open(path, 'w') as f:

            if isinstance(self.method.data, np.ndarray):
            
                f.write(self.method.comment + "\n")

                for row in self.method.data:
                    data_line = (' '.join(["{: 12.6f}" * len(row)])).format( *tuple(row) ) + "\n"
                    f.write(data_line)
            elif isinstance(self.method.data, str):
                f.write(self.method.data)
        
            

        


if __name__ == '__main__':
    # Test out the functionality

    plot=0 #1
    original_image_directory = "images"    
    image_directory = "images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_RandomWalkerSegmentation"

    image_directory = "images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_OtsuThreshold"
    image_directory = "images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"

    image_directory = "images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_RandomWalkerSegmentation"
    image_directory = "images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"
    #    image_directory = "images_imageJ_analysed"
    print(f"Analysing images from {image_directory}..")    
    # Now analyse
    print(f"> Alpha-beta fraction, profile")
    analysis = AlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory, original_image_directory)
    ac.analyse_images(plot=plot)

    # print(f"> Chris Alpha-beta fraction")
    # analysis = ChrisAlphaBetaFraction
    # ac = AnalysisContainer(analysis, image_directory, original_image_directory)
    # ac.analyse_images(plot=plot)
