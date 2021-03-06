import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
from typing import Type, TypeVar
from datetime import datetime
from skimage import color, exposure, img_as_float, img_as_uint, img_as_ubyte, util, measure
import imageio
from scipy.optimize import curve_fit, minimize

# Make an abstract base class which supplies a processing function upon an image
class DataAnalysis(ABC):
    
    @abstractmethod
    def analyse(self, image):
        pass
            
    @abstractmethod
    def plot(self, image):
        pass

    
class SegmentData:
    def __init__(self, data, line):
        self.data = data
        self.line = line
    
    def segment_from_condition(self, cond):
        i = np.arange(len(cond))[cond]

        cond_data = np.zeros((len(i), self.data.shape[1]))
        cond_data[:,0] =  (self.data[:,0])[i]
        cond_data[:,1] =  (self.data[:,1])[i]

        return cond_data

    def segment_data(self):
        # Segment data naively, to classify the data into two sets,
        # such that one can compare both.

        before = self.data[:,0] <  self.line
        after  = self.data[:,0] >= self.line

        before_data = self.segment_from_condition( before)
        after_data  = self.segment_from_condition( after )

        # print("before data = ", before_data)
        # print("after data = ", after_data)        

        return before_data, after_data


   
class LinearFit:
    def flat_line(self, x, c):
        return c

    def straight_line(self, x, m, c):
        return m*x + c

    def exponential_line(self, x, A, b, c):
        return A * np.exp( -b * x ) + c
    
    def fit_curve(self, data, expected, func):
        x, y = data[:,0], data[:,1]
        params,cov=curve_fit(func, x, y, expected, maxfev=1000)
        stdevs = np.sqrt(np.diag(cov))
        return params, stdevs        

    def flat(self, data):
        d = { "data" : data,
              "name" : "flat line",
              "func" : self.flat_line,
              "expected" : (0.7,)}
        return d

    def straight(self, data, **kwargs):
        d = { "data" : data,
              "name" : "straight line",
              "func" : self.straight_line,
              "expected" : (-1/2500., 0.7)}

        for k,v in kwargs.items():
            if k in d:
                d[k] = v
        print(d["expected"])
        return d
    
    def exponential(self, data):
        d = { "data" : data,
              "name" : "exponential line",
              "func" : self.exponential_line,
              "expected" : (5., 0.005, 0.7)}
        return d
    

    def fit(self, data=None, func=None, expected=None, name=""):
        try:
           params, stdevs = self.fit_curve(data, expected, func) # self.exp_analysis()
        except ValueError:
            print(f"{name} fit: WARNING: Cannot fit to {name}, will give flat dependence for line")
            params = expected
            stdevs = np.array([1 for i in params])
        except RuntimeError:
            print(f"{name} fit: WARNING: Cannot fit to {name} after many iterations, will give flat dependence")
            params = (0., 0.000, 0.7)
            stdevs = np.array([1 for i in params])

        return params, stdevs
    

    
class BisectionFit(DataAnalysis):
    def __init__(self):
        self.fit = LinearFit()
        self.name = "BisectionFit"
        self.comment = "# Gradient_from_surface[ab_frac/pixel] mean_in_bulk[ab_frac] surface_bulk_ratio"

    def compare_methods(self, linear_data, flat_data):
        self.params_flat,     self.stdevs_flat     = self.fit.fit(**self.fit.flat(flat_data))
        self.params_straight, self.stdevs_straight = self.fit.fit(**self.fit.straight(linear_data))

        # compare the standard deviation between the points and see what fits better
        # Extent of data before
        mean_fit = (  self.fit.straight_line( self.before[ 0,0], self.params_straight[0], self.params_straight[1])
                    + self.fit.straight_line( self.before[-1,0], self.params_straight[0], self.params_straight[1]) ) / 2
        f_data = self.params_flat[0], self.stdevs_flat[0]
        s_data_g, s_data_i = (self.params_straight[0], self.stdevs_straight[0]), (self.params_straight[1], self.stdevs_straight[1])

        # self.data = "{self.params_straight[0]} {self.stdevs_straight[0]} {self.params_straight[1]} {self.stdevs_straight[0]} {self.params_flat[0]} {self.stdevs_flat[0]} \n"

        self.data = f"{self.params_straight[0]} {self.params_flat[0]} {mean_fit/self.params_flat[0]}\n"
        return f_data, s_data_g, s_data_i
        

    def analyse(self, data):
        self.original_data = data
        # Here we test the hypotheses, based on the data we botain
        # Give initial line
        line = (np.max(self.original_data[:,0]) - np.min(self.original_data[:,0])) / 2.
        
        segment = SegmentData(self.original_data, line)
        self.before, self.after = segment.segment_data()

        # On the segmented data I could do cross validation to find
        # the best model and error. This can determine the threshold.

        return self.compare_methods(self.before, self.after)


    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].plot(self.original_data[:,0], self.original_data[:,1])

        yb  = self.fit.straight_line(self.before[:,0], self.params_straight[0], self.params_straight[1])
        yb1 = self.fit.straight_line(self.before[:,0], self.params_straight[0] + self.stdevs_straight[0], self.params_straight[1] + self.stdevs_straight[1])
        yb2 = self.fit.straight_line(self.before[:,0], self.params_straight[0] - self.stdevs_straight[0], self.params_straight[1] - self.stdevs_straight[1])


        ya  = np.array( [ self.fit.flat_line(val, self.params_flat[0])                       for val in self.after[:,0] ] )
        ya1 = np.array( [ self.fit.flat_line(val, self.params_flat[0] + self.stdevs_flat[0]) for val in self.after[:,0] ] )
        ya2 = np.array( [ self.fit.flat_line(val, self.params_flat[0] - self.stdevs_flat[0]) for val in self.after[:,0] ] )

        
        ax[1].plot(self.before[:,0], yb, 'r-')
        ax[1].plot(self.before[:,0], yb1, 'g--')
        ax[1].plot(self.before[:,0], yb2, 'g--')
        ax[1].fill_between(self.before[:,0], yb1, yb2, facecolor="gray", alpha=0.15)

        print("Segmented data: ",self.before.shape, self.after.shape)
        ax[1].plot(self.after[:,0], ya, 'b-')
        ax[1].plot(self.after[:,0], ya1, 'm--')
        ax[1].plot(self.after[:,0], ya2, 'm--')
        ax[1].fill_between(self.after[:,0], ya1, ya2, facecolor="gray", alpha=0.15)

        
        ax[1].set_xlabel("Surface Depth")
        ax[1].set_ylabel("Alpha-beta fraction")        
        ax[1].set_title('Alpha-beta volume fraction with surface depth')
        fig.tight_layout()
        plt.show()
        
        
class MultisectionFit(DataAnalysis):
    def __init__(self):
        self.fit = LinearFit()
        self.name = "MultisectionFit"
        self.comment = "# surface_bulk_ratio"

    def compare_methods(self, linear_data, flat_data):
        self.params_flat,     self.stdevs_flat     = self.fit.fit(**self.fit.flat(flat_data))
        #        self.params_exponential, self.stdevs_exponential = self.fit.fit(**self.fit.exponential(linear_data))
        self.params_exponential, self.stdevs_exponential = self.fit.fit(**self.fit.straight(linear_data))
        # intercept = np.min(linear_data[:,1])
        # log_linear = np.zeros(linear_data.shape)
        # log_linear[:,0] = linear_data[:,0]
        # log_linear[:,1] = np.log(linear_data[:,1] - intercept)
        # self.params_exponential, self.stdevs_exponential = self.fit.fit(**self.fit.straight(log_linear, expected = (- np.max(log_linear[:,1])/2500., np.max(log_linear[:,1]))))
        # # straight line has parmeters in order m, c, when exp would be np.exp(c) np.exp()
        # self.params_exponential =  (np.exp(self.params_exponential[1]), self.params_exponential[0], intercept )
        # self.stdevs_exponential = (np.exp(self.stdevs_exponential[1]), self.stdevs_exponential[0], np.std(linear_data[:,1]))
        
        # self.params_straight, self.stdevs_straight = self.fit.fit(**self.fit.straight(linear_data))

        # compare the standard deviation between the points and see what fits better
        # Extent of data before
        # mean_fit = (  self.fit.straight_line( self.before[ 0,0], self.params_straight[0], self.params_straight[1])
        #             + self.fit.straight_line( self.before[-1,0], self.params_straight[0], self.params_straight[1]) ) / 2

        line = (np.max(flat_data[:,0]) + np.min(flat_data[:,0])) / 2.
        segment2 = SegmentData(flat_data, line)
        before, after = segment2.segment_data()
        
        mean_fit = np.mean(self.before[:,1])
        mean_end = np.mean(after[:,1])
        f_data = self.params_flat[0], self.stdevs_flat[0]
        s_data_g, s_data_i = (self.params_exponential[0], self.stdevs_exponential[0]), (self.params_exponential[1], self.stdevs_exponential[1])


        print("SB ratio: ", mean_fit/mean_end)
        
        # self.data = "{self.params_straight[0]} {self.stdevs_straight[0]} {self.params_straight[1]} {self.stdevs_straight[0]} {self.params_flat[0]} {self.stdevs_flat[0]} \n"

        #        self.data = f"{self.params_straight[0]} {self.params_flat[0]} {mean_fit/self.params_flat[0]}\n"
        self.data = f" {mean_fit/mean_end}\n"        
        return f_data, s_data_g, s_data_i
        

    def analyse(self, data):
        self.original_data = data
        # Here we test the hypotheses, based on the data we botain
        # Give initial line
        line = (np.max(self.original_data[:,0]) + np.min(self.original_data[:,0])) / 2.
        
        segment = SegmentData(self.original_data, line)
        self.before, self.after = segment.segment_data()

        n_partition = 6 # 3 means data is split into 2^3 = 8ths
        for i in range(n_partition):
            line = (np.max(self.before[:,0]) + np.min(self.before[:,0])) / 2.
            segment = SegmentData(self.before, line)
            self.before, self.before2 = segment.segment_data()
        
        for i in range(1):
            line = (np.max(self.after[:,0]) + np.min(self.after[:,0])) / 2.
            segment = SegmentData(self.after, line)
            self.after1, self.after = segment.segment_data()
        
        
        # On the segmented data I could do cross validation to find
        # the best model and error. This can determine the threshold.

        return self.compare_methods(self.before, self.after)


    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].scatter(self.original_data[:,0], self.original_data[:,1])

        # yb  = self.fit.exponential_line(self.before[:,0], self.params_exponential[0], self.params_exponential[1], self.params_exponential[2])
        # yb1 = self.fit.exponential_line(self.before[:,0],
        #                                 self.params_exponential[0] + self.stdevs_exponential[0],
        #                                 self.params_exponential[1] - self.stdevs_exponential[1],
        #                                 self.params_exponential[2] - self.stdevs_exponential[2])
        # yb2 = self.fit.exponential_line(self.before[:,0],
        #                                 self.params_exponential[0] - self.stdevs_exponential[0],
        #                                 self.params_exponential[1] + self.stdevs_exponential[1],
        #                                 self.params_exponential[2] + self.stdevs_exponential[2])


        yb  = self.fit.straight_line(self.before[:,0], self.params_exponential[0], self.params_exponential[1])
        yb1 = self.fit.straight_line(self.before[:,0],
                                        self.params_exponential[0] + self.stdevs_exponential[0],
                                        self.params_exponential[1] - self.stdevs_exponential[1]
                                        )
        yb2 = self.fit.straight_line(self.before[:,0],
                                        self.params_exponential[0] - self.stdevs_exponential[0],
                                        self.params_exponential[1] + self.stdevs_exponential[1])

        ya  = np.array( [ self.fit.flat_line(val, self.params_flat[0])                       for val in self.after[:,0] ] )
        ya1 = np.array( [ self.fit.flat_line(val, self.params_flat[0] + self.stdevs_flat[0]) for val in self.after[:,0] ] )
        ya2 = np.array( [ self.fit.flat_line(val, self.params_flat[0] - self.stdevs_flat[0]) for val in self.after[:,0] ] )

        
        ax[1].plot(self.before[:,0], yb, 'r-', label="Denudation fit")
        ax[1].plot(self.before[:,0], yb1, 'g--')
        ax[1].plot(self.before[:,0], yb2, 'g--')
        ax[1].fill_between(self.before[:,0], yb1, yb2, facecolor="gray", alpha=0.15)

        print("Segmented data: ",self.before.shape, self.after.shape)
        ax[1].plot(self.after[:,0], ya, 'b-', label = "Bulk mean")
        ax[1].plot(self.after[:,0], ya1, 'm--')
        ax[1].plot(self.after[:,0], ya2, 'm--')
        ax[1].fill_between(self.after[:,0], ya1, ya2, facecolor="gray", alpha=0.15)

        
        ax[1].set_xlabel("Surface Depth")
        ax[1].set_ylabel("Alpha-beta fraction")        
        ax[1].set_title('Alpha-beta volume fraction vs depth')
        ax[1].legend()
        fig.tight_layout()
        plt.show()
        
        

        
    
    
class PreprocessContainer:
    def __init__(self, analysis_method: Type[DataAnalysis], image_directory: str, data_directory: str): #, denudation_data:str):

        # self.denudation_images  = np.loadtxt(denudation_data, usecols=(0,), skiprows=1, dtype=np.str)
        # self.denudation_classification  = np.loadtxt(denudation_data, usecols=(1,), skiprows=1, dtype=np.int)        

        self.image_directory = image_directory
        self.images = sorted(os.listdir(self.image_directory)) # self.sort_files_by_data(sorted(os.listdir(self.image_directory)), self.denudation_images)

        self.data_directory = data_directory
        self.data_files = sorted(os.listdir(self.data_directory)) # self.sort_analysis_files_by_data( sorted(os.listdir(self.data_directory)), self.denudation_images)
        
        self.method = analysis_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")

    def analyse_images(self, plot=True):
        mode = 'w'

        for data_file, image in zip(self.data_files, self.images):
            unprocessed_name = image #self.get_filename_from_id(image)
            print(f"\ndata_file = {data_file}\nimage = {self.image_directory}/{unprocessed_name} \n ")
            unprocessed_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{unprocessed_name}"))

            # This gives the pixel depth, alpha beta volume fraction data
            data = np.loadtxt(f"{self.data_directory}/{data_file}", usecols=(0,1), skiprows=1)

            self.method.analyse( data )
            if plot:
                self.method.plot(unprocessed_image)

            self.method.comment.replace("#", "# Image ")
            self.save(image, f"{image}  " + self.method.data, mode)
            mode='a'

    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.dt}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, data, mode='w'):
        # self.dir_name = self.create_output_directory()
        self.dir_name = ''
        name = os.path.splitext(image_name)[0]
        path = os.path.join(self.dir_name, f"{self.method_name}_{self.dt}.dat")
        
        self.path = path
        print(f"> Saving data to {path}")
        with open(path, mode) as f:

            if isinstance(self.method.data, np.ndarray):
            
                f.write(self.method.comment + "\n")

                for row in self.method.data:
                    data_line = (' '.join(["{: 12.6f}" * len(row)])).format( *tuple(row) ) + "\n"
                    f.write(data_line)
            elif isinstance(self.method.data, str):
                if mode == 'w':
                    f.write(self.method.comment + "\n")
                f.write(data)

    def sort_files_by_data(self, files, ordered_list, condition_func=None):
        names = []
        for o in ordered_list:
            for f in files:
                if o in f:
                    names.append(f)
        print("sort files by data: ", len(names), len(ordered_list), len(files))
        return names

    def sort_analysis_files_by_data(self, files, ordered_list, condition_func=None):
        names = []
        for o in ordered_list:
            for f in files:
                base_name = os.path.splitext(f.split('_')[1])[0]
                if base_name == o:
                    names.append(f)
        print("sort analysis files", len(names), len(ordered_list), len(files))
        return names


    def get_filename_from_id(self, name):
        analysis_name = os.path.splitext(name.split('_')[0])[0]
        base_name = os.path.splitext(name.split('_')[1])[0]
        extension = os.path.splitext(name)[1]
        prefixed = [filename for filename in sorted(os.listdir(self.image_directory)) if filename.startswith(base_name)]

        print("Unprocessed image: ", prefixed[0])
        return prefixed[0]

if __name__ == "__main__":

    plot=0 # False #True
    image_directory =  "images" #"images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"
    data_directory = "AlphaBetaFraction_2021-11-23--12-44-13_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_OtsuThreshold"
    # data_directory = "AlphaBetaFraction_2021-11-24--16-05-20_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_RandomWalkerSegmentation"

    # data_directory="AlphaBetaFraction_2021-11-24--17-02-36_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_RandomWalkerSegmentation"

    data_directory = "AlphaBetaFraction_2021-11-26--12-11-12_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_OtsuThreshold"

    
    data_directory = "AlphaBetaFraction_2021-11-29--09-24-21_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"

    data_directory = "AlphaBetaFraction_2021-11-29--11-55-29_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"
    data_directory = "AlphaBetaFraction_2021-11-29--14-56-37_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"

    data_directory = "AlphaBetaFraction_2021-11-30--14-51-43_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"
    denudation_data = "denudation_data_new.dat"

    data_directory = "AlphaBetaFraction_2021-12-03--13-16-32_new_images_RemoveBakeliteBoundary_WhiteBackgroundRemoval_HistogramEquilization_OtsuThreshold"
    
    print(f"Preprocessing data from {data_directory}, with images from {image_directory}..")
    # Now analyse
    print(f"> Gradient data from alpha-beta fraction")
    analysis =  MultisectionFit #BisectionFit
    ac = PreprocessContainer(analysis, image_directory, data_directory)
    ac.analyse_images(plot=plot)

    # print(f"> Chris Alpha-beta fraction")
    # analysis = ChrisAlphaBetaFraction
    # ac = AnalysisContainer(analysis, image_directory)
    # ac.analyse_images(plot=plot)
