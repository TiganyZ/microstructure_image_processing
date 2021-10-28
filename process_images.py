"""This is a file which implements multiiple types of image processing
to microstructures such that one can perform analysis

--- Structure ---

> Will have a class which takes the input, which is a directory of
  images, or an image.

> For each step in the processing of an image, we will make a
  directory, which has the process name and the corresponding image
  inside with a given name as extension.

> It makes sense to use a strategy pattern, from this, one can just
  have a class which takes a directory of images, and then applies a
  given function to them, which then spits out the images

> The first thing to do will be to make the initial results directory
  which has the input images and then the corresponding outputs.


"""
import matplotlib.pyplot as plt
import imageio
import numpy as np
from copy import copy, deepcopy
from skimage import  color, data, restoration, exposure, img_as_float, img_as_uint, img_as_ubyte, util
from skimage.filters import threshold_otsu

from functools import partial

from scipy.optimize import curve_fit, minimize
from scipy import ndimage
from scipy.signal import convolve2d as conv2

from datetime import datetime

from abc import ABC, abstractmethod
import argparse
import os
from typing import Type, TypeVar


        

# Make an abstract base class which supplies a processing function upon an image
class ProcessImage(ABC):
    
    @abstractmethod
    def process(self):
        pass
            
    @abstractmethod
    def plot(self, image):
        pass

# TProcessImage = TypeVar("TProcessImage", bound=ProcessImage)

####################################################################
###---   Thresholding - Using Otsu's automatic thresholding   ---###
####################################################################

# Interesting automatic thresholding technique, analyse for caveats
# > Wikipedia article https://en.wikipedia.org/wiki/Otsu's_method#Limitations
# > Works well for a bimodal distribution of pixel intensities.
    
class OtsuThreshold(ProcessImage):
    def __init__(self):
        self.name = "OtsuThreshold"
        
    def process(self, image):
        self.thresh = threshold_otsu(image)
        self.output = img_as_ubyte(image > self.thresh)

        
    def plot(self, original_image):
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        ax[0].imshow(original_image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].hist(original_image.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(self.thresh, color='r')

        ax[2].imshow(self.output, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        plt.show()

class Threshold(ProcessImage):
    def __init__(self):
        self.name = "Threshold"

    def gauss(self, x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    def bimodal(self, x,mu1,sigma1,A1,mu2,sigma2,A2):
        return self.gauss(x,mu1,sigma1,A1)+self.gauss(x,mu2,sigma2,A2)

    def quadmodal(self, x,mu1,sigma1,A1,mu2,sigma2,A2,
                    mu3,sigma3,A3,mu4,sigma4,A4):
        return (gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
                + gauss(x,mu3,sigma3,A3)+gauss(x,mu3,sigma3,A3) )

    def max_argument_from_condition(self, cond, x, y):
        xind = np.arange(len(x))[cond]
        arg = xind[ np.argsort(y[cond])[-1] ]
        return arg
    
    def fit_bimodal(self, image):
        print("Fitting bimodal distribution to the pixel intensities")
        intensity = image.ravel()
        # Expect peaks at the bottom and top quartile
        # > This is probably dependent on image size for the height
        y, x, _ = plt.hist(intensity, bins=256)
        x = x[:-1]
        
        # histogram, bin_edges = np.histogram(image, bins=256, range=(0, 1))

        # Find the two highest modes of the distribution and then like to the intensities and means of the gaussians

        argm1 = self.max_argument_from_condition( x<0.4, x, y)
        argm2 = self.max_argument_from_condition( x>0.4, x, y)        
        
        xm1, xm2 = x[argm1], x[argm2]
        ym1, ym2 = y[argm1], y[argm2]
        sm1, sm2 = 0.1, 0.2
        
        expected =  (xm1, sm1, ym1,   xm2, sm2, ym2)
        print(" > expected x, mu, sig, for each gaussian: ", expected[:3], "  ", expected[3:])
        
        plt.plot(x, partial(self.bimodal, mu1=xm1,sigma1=sm1,A1=ym1, mu2=xm2,sigma2=sm2,A2=ym2)(x)  )
        plt.show()
        
        params,cov=curve_fit(self.bimodal, x, y, expected)

        mu1,sigma1,A1,mu2,sigma2,A2 = params
        # Now we can find the minimum between the two peaks.
        # > Maybe we can actually do a general gaussian mixture and determin the number later!
        # > This is convoluted, and would likely vary for other microstructural images.

        # We can find the minimum by averaging between the means and using our favourite algorithm.
        x0 = np.array([(mu1 + mu2)/2.])
        fnc = partial(self.bimodal, mu1=mu1,sigma1=sigma1,A1=A1, mu2=mu2,sigma2=sigma2,A2=A2)

        plt.hist(intensity, bins=256, label = "Pixel Intensity")
        plt.plot(x, fnc(x)  , label="Bimodal Fit")
        plt.legend()
        plt.show()
        

        print("> Checking partial function works:", params)
        print("> self.bimodal(*( (0.0,) + expected)) = ", self.bimodal(*( (0.0,) + tuple(params))))
        print("> partial(0.0) = ", fnc(0.0))
        
        method = 'Nelder-Mead'
        print(f"> Minimising partial bimodal function with {method}")
        ret =  minimize(fnc, x0, method=method, options={'disp': True, 'fatol': 1e-4})

        return ret['x'][0], ret['fun']


    
    def process(self, image):
        # Manual threshold 
        self.thresh = 0.5
        self.output = img_as_ubyte(image > self.thresh)

        # One can run a gaussian deconvolution to find which peaks correspond to which phase
        

    def plot(self, original_image):
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        ax[0].imshow(original_image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].hist(original_image.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(self.thresh, color='r')

        ax[2].imshow(self.output, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        plt.show()
        
        
class RemoveBakelite(ProcessImage):
    def __init__(self):
        self.name = "RemoveBakelite"

    def gauss(self, x,mu,sigma,A):
        return A*np.exp(-(x-mu)**2/2/sigma**2)

    def bimodal(self, x,mu1,sigma1,A1,mu2,sigma2,A2):
        return self.gauss(x,mu1,sigma1,A1)+self.gauss(x,mu2,sigma2,A2)

    def max_argument_from_condition(self, cond, x, y):
        xind = np.arange(len(x))[cond]
        arg = xind[ np.argsort(y[cond])[-1] ]
        return arg
    
    def fit_bimodal(self, image):
        print("Fitting bimodal distribution to the pixel intensities")
        intensity = image.ravel()
        # Expect peaks at the bottom and top quartile
        # > This is probably dependent on image size for the height
        y, x, _ = plt.hist(intensity, bins=256)
        x = x[:-1]
        
        # Find the two highest modes of the distribution and then like to the intensities and means of the gaussians

        argm1 = self.max_argument_from_condition( x<0.4, x, y)
        argm2 = self.max_argument_from_condition( x>0.4, x, y)        
        
        xm1, xm2 = x[argm1], x[argm2]
        ym1, ym2 = y[argm1], y[argm2]
        sm1, sm2 = 0.1, 0.2
        
        expected =  (xm1, sm1, ym1,   xm2, sm2, ym2)
        
        params,cov=curve_fit(self.bimodal, x, y, expected)

        mu1,sigma1,A1,mu2,sigma2,A2 = params
                
        x0 = np.array([(mu1 + mu2)/2.])
        fnc = partial(self.bimodal, mu1=mu1,sigma1=sigma1,A1=A1, mu2=mu2,sigma2=sigma2,A2=A2)
        method = 'Nelder-Mead'
        ret =  minimize(fnc, x0, method=method, options={'disp': True, 'fatol': 1e-4})

        return ret['x'][0], ret['fun']

    def remove_bakelite_pixels(self, image, thresholded_image, plot=False):
        # Take only the bakelite from the top of the image
        # This function takes in a thresholded image and then iterates over the top of the array
        new_image = deepcopy(image)
        deleted = 0
        for i,row in enumerate(thresholded_image):
            if i > 1/3. * len(thresholded_image):
                break

            # Puttimg threshold here for the mean values of pixels, if 90% are black then remove
            if np.mean(row) < 0.1:
                # Then modify image by deleting the row
                new_image = np.delete(new_image, i-deleted, 0)
                deleted += 1
        if plot:
            fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
            ax = axes.ravel()
            ax[0] = plt.subplot(1, 3, 1)
            ax[1] = plt.subplot(1, 3, 2)
            ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

            ax[0].imshow(image, cmap=plt.cm.gray)
            ax[0].set_title('Original')
            ax[0].axis('off')

            ax[1].imshow(thresholded_image, cmap=plt.cm.gray)
            ax[1].set_title('Thresholded')
            ax[1].axis('off')

            ax[2].imshow(new_image, cmap=plt.cm.gray)
            ax[2].set_title('Removed Bakelite')

            plt.show()
        return new_image
    
    def process(self, image):
        # Now the optimal threshold is the local minima between the
        # bimodal distribution found in the histogram

        # We can do this by finding each of the maxima and then using
        # a minimisation algorithm

        # First fit a bimodal distribution to the image histogram

        self.thresh, value = self.fit_bimodal(image)

        # The bottom gaussian corresponds to the black bakelite, therefore remove
        self.output = self.remove_bakelite_pixels(image, img_as_float(image > self.thresh))
        

    def plot(self, original_image):
        fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
        ax = axes.ravel()
        ax[0] = plt.subplot(1, 3, 1)
        ax[1] = plt.subplot(1, 3, 2)
        ax[2] = plt.subplot(1, 3, 3, sharex=ax[0], sharey=ax[0])

        ax[0].imshow(original_image, cmap=plt.cm.gray)
        ax[0].set_title('Original')
        ax[0].axis('off')

        ax[1].hist(original_image.ravel(), bins=256)
        ax[1].set_title('Histogram')
        ax[1].axvline(self.thresh, color='r')

        ax[2].imshow(self.output, cmap=plt.cm.gray)
        ax[2].set_title('Thresholded')
        ax[2].axis('off')

        plt.show()
        
        

#####################################
###---   Image deconvolution   ---###
#####################################

class DeconvoluteNoise(ProcessImage):
    def __init__(self):
        self.name = "DeconvoluteNoise"
        
    def process(self, image):
        # Define a point spread function, this one is arbitrary
        psf = np.ones((5, 5)) / 25
        #        self.output = conv2(image, psf, 'same')
        self.output = restoration.richardson_lucy(image, psf, 30)

        
    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 5))
        plt.gray()

        for a in (ax[0], ax[1]):
               a.axis('off')

        ax[0].imshow(original_image)
        ax[0].set_title('Original Data')

        ax[1].imshow(self.output, vmin=original_image.min(), vmax=original_image.max())
        ax[1].set_title('Restoration using\nRichardson-Lucy')


        fig.subplots_adjust(wspace=0.02, hspace=0.2,
                            top=0.9, bottom=0.05, left=0, right=1)
        plt.show()

    


########################################
###---   Histogram Equilization   ---###
########################################
# https://en.wikipedia.org/wiki/Histogram_equalization

class HistogramEquilization(ProcessImage):
    def __init__(self):
        self.name = "HistogramEquilization"

    def process(self, image):
        # Contrast stretching
        p2, p98 = np.percentile(image, (2, 98))
        self.image_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

        # Equalization
        self.image_eq = exposure.equalize_hist(image)
        
        # Adaptive Equalization
        self.image_adapteq = exposure.equalize_adapthist(image, clip_limit=0.03)
        self.output = self.image_adapteq
        
        

    def plot_img_and_hist(self, image, axes, bins=256):
        """Plot an image along with its histogram and cumulative histogram.

        """
        image = img_as_float(image)
        ax_image, ax_hist = axes
        ax_cdf = ax_hist.twinx()

        # Display image
        ax_image.imshow(image, cmap=plt.cm.gray)
        ax_image.set_axis_off()

        # Display histogram
        ax_hist.hist(image.ravel(), bins=bins, histtype='step', color='black')
        ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
        ax_hist.set_xlabel('Pixel intensity')
        ax_hist.set_xlim(0, 1)
        ax_hist.set_yticks([])

        # Display cumulative distribution
        image_cdf, bins = exposure.cumulative_distribution(image, bins)
        ax_cdf.plot(bins, image_cdf, 'r')
        ax_cdf.set_yticks([])

        return ax_image, ax_hist, ax_cdf

    def plot(self, image):
        # Display results
        fig = plt.figure(figsize=(8, 5))
        axes = np.zeros((2, 4), dtype=np.object)
        axes[0, 0] = fig.add_subplot(2, 4, 1)
        for i in range(1, 4):
            axes[0, i] = fig.add_subplot(2, 4, 1+i, sharex=axes[0,0], sharey=axes[0,0])
        for i in range(0, 4):
            axes[1, i] = fig.add_subplot(2, 4, 5+i)

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(image, axes[:, 0])
        ax_img.set_title('Low contrast image')

        y_min, y_max = ax_hist.get_ylim()
        ax_hist.set_ylabel('Number of pixels')
        ax_hist.set_yticks(np.linspace(0, y_max, 5))

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(self.image_rescale, axes[:, 1])
        ax_img.set_title('Contrast stretching')

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(self.image_eq, axes[:, 2])
        ax_img.set_title('Histogram equalization')

        ax_img, ax_hist, ax_cdf = self.plot_img_and_hist(self.image_adapteq, axes[:, 3])
        ax_img.set_title('Adaptive equalization')

        ax_cdf.set_ylabel('Fraction of total intensity')
        ax_cdf.set_yticks(np.linspace(0, 1, 5))

        # prevent overlap of y-axis labels
        fig.tight_layout()
        plt.show()




#######################################################################
###---   FFT Gaussian filter -- for ease of use and generality   ---###
#######################################################################

class FFTGaussianFilter(ProcessImage):
    def __init__(self):
        self.name = "FFTGaussianFilter"

    def process(self, image):
        sigma = 4
        self.output = ndimage.gaussian_filter(image, sigma)

    def plot(self, original_image):
        plt.figure()
        plt.imshow(self.output, plt.cm.gray)
        plt.title('Blurred image with Gaussian filter')
        plt.show()


##########################################
###---   White Background Removal   ---###
##########################################

class WhiteBackgroundRemoval(ProcessImage):
    def __init__(self):
        self.name = "WhiteBackgroundRemoval"

    def process(self, image):
        image_inverted = util.invert(image)
        background_inverted = restoration.rolling_ball(image_inverted, radius=90)

        self.output = util.invert(image_inverted - background_inverted)
        self.background = util.invert(background_inverted)

    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].imshow(self.background, cmap='gray')
        ax[1].set_title('Background')
        ax[1].axis('off')

        ax[2].imshow(self.output, cmap='gray')
        ax[2].set_title('Result')
        ax[2].axis('off')

        fig.tight_layout()

        plt.show()


##########################################
###---   Black Background Removal   ---###
##########################################


class BlackBackgroundRemoval(ProcessImage):
    def __init__(self):
        self.name = "BlackBackgroundRemoval"

    def process(self, image):
        self.background = restoration.rolling_ball(image, radius=45)
        self.output = image - self.background
        

    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=3)

        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].imshow(self.background, cmap='gray')
        ax[1].set_title('Background')
        ax[1].axis('off')

        ax[2].imshow(self.output, cmap='gray')
        ax[2].set_title('Result')
        ax[2].axis('off')

        fig.tight_layout()

        plt.show()

 


# Make a containing class which takes a process image instance and
# then makes a corresponding directory for it

class ProcessContainer:
    def __init__(self, processing_method: Type[ProcessImage], image_directory: str):
        self.image_directory = image_directory
        self.images = os.listdir(self.image_directory)

        self.method = processing_method()
        self.method_name = self.method.name


    def process_images(self, plot=True):
        for image in self.images:
            original_image =  color.rgb2gray(imageio.imread(f"{self.image_directory}/{image}"))
            self.method.process( original_image )
            if plot:
                self.method.plot(original_image)
            self.save(image, self.method.output)


    def create_output_directory(self):
        # now = datetime.now()
        # dt = now.strftime("%Y-%m-%d--%H-%M-%S")

        dir_name = f"{self.method_name}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, image):
        self.dir_name = self.create_output_directory()
        path = os.path.join(self.dir_name, f"{self.method_name}_{image_name}")
        
        print(f"> Saving processed image to {path}")
        imageio.imwrite(path, img_as_ubyte(image))




        
if __name__ == '__main__':
    # Test out the functionality

    image_directory = "images"

    processes = [OtsuThreshold,
                 Threshold,
                 DeconvoluteNoise,
                 HistogramEquilization,
                 FFTGaussianFilter,
                 BlackBackgroundRemoval,
                 WhiteBackgroundRemoval]

    processes = [RemoveBakelite, WhiteBackgroundRemoval,  OtsuThreshold]
    for process in processes:
        pc = ProcessContainer(process, image_directory)
        pc.process_images(plot=True)

        image_directory = pc.dir_name
    
    
    

        

# ################################################
# ###---   Thresholding - try all of them   ---###
# ################################################

# from skimage.filters import try_all_threshold

# img = data.page()

# fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
# plt.show()



