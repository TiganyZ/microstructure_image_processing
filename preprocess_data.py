import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
from typing import Type, TypeVar

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

        cond_data = np.zeros((len(cond), self.data.shape[1]))
        cond_data[:,0] =  (self.data[:,0])[cond]
        cond_data[:,1] =  (self.data[:,1])[cond]

        return cond_data

    def segment_data(self):
        # Segment data naively, to classify the data into two sets,
        # such that one can compare both.

        before = self.data[:,1] <  self.line
        after  = self.data[:,1] >= self.line

        before_data = self.segment_from_condition(self.data, before)
        after_data  = self.segment_from_condition(self.data, after )        

        return before_data, after_data

    
class LinearFit:
    def flat_line(self, x, c):
        return c

    def straight_line(self, x, m, c):
        return m*x + c
    
    def fit_curve(self, data, expected, func):
        x, y = data[:,0], data[:,1]
        params,cov=curve_fit(func, x, y, expected)
        stdevs = np.sqrt(np.diag(cov))
        print(params)
        return params, stdevs        

    def flat(self, data):
        d = { "data" : data,
              "name" : "flat line"
              "func" : self.flat_line,
              "expected" : (0.7,)}
        return d

    def straight(self, data):
        d = { "data" : data,
              "name" : "straight line"
              "func" : self.straight_line,
              "expected" : (-1/2500., 0.7)}
        return d
    

    def fit(self, data=None, func=None, expected=None, name=""):
        try:
           params, stdevs = self.fit_curve(data, expected, func) # self.exp_analysis()
        except ValueError:
            print(f"{name} fit: WARNING: Cannot fit to {name}, will give flat dependence for line")
            params = expected
            stdevs = np.array([1 for i in params])
        return params, stdevs
    

    
class FitSegmentData(DataAnalysis):
    def __init__(self, data):
        self.fit = LinearFit()
        self.data = data
    
    def compare_methods(self, linear_data, flat_data):
        self.params_flat,     self.stdevs_flat     = self.fit(**self.flat(flat_data))
        self.params_straight, self.stdevs_straight = self.fit(**self.straight(linear_data))

        # compare the standard deviation between the points and see what fits better
        f_data = self.params_flat[0], self.stdevs_flat[0]
        s_data_g, s_data_i = (self.params_straight[0], self.stdevs_straight[0]) (self.params_straight[1], self.stdevs_straight[1])

        self.output = "{self.params_straight[0]} {self.stdevs_straight[0]} {self.params_straight[1]} {self.stdevs_straight[0]} {self.params_flat[0]} {self.stdevs_flat[0]} \n"
        return f_data, s_data_g, s_data_i
        

    def analyse(self):
        # Here we test the hypotheses, based on the data we botain
        # Give initial line
        line = (np.max(self.data[:,0]) - np.min(self.data[:,0])) / 2.
        
        segment = SegmentData(self.data, line)
        self.before, self.after = segment.segment_data()

        # On the segmented data I could do cross validation to find
        # the best model and error. This can determine the threshold.

        return compare_methods(self.before, self.after)


    def plot(self, original_image):
        fig, ax = plt.subplots(nrows=1, ncols=2)

        ax[0].imshow(original_image, cmap='gray')
        ax[0].set_title('Original image')
        ax[0].axis('off')

        ax[1].plot(self.data[:,0], self.data[:,1])


        yb  = self.fit.straight_line(self.before[:,0], self.params_straight[0], self.params_straight[1])
        yb1 = self.fit.straight_line(self.before[:,0], self.params_straight[0] + self.stdevs_straight[0], self.params_straight[1] - self.stdevs_straight[1])
        yb2 = self.fit.straight_line(self.before[:,0], self.params_straight[0] - self.stdevs_straight[0], self.params_straight[1] + self.stdevs_straight[1])


        ya  = self.fit.flat_line(self.after[:,0], self.params_flat[0], self.params_flat[1])
        ya1 = self.fit.flat_line(self.after[:,0], self.params_flat[0] + self.stdevs_flat[0], self.params_flat[1] - self.stdevs_flat[1])
        ya2 = self.fit.flat_line(self.after[:,0], self.params_flat[0] - self.stdevs_flat[0], self.params_flat[1] + self.stdevs_flat[1])

        
        ax[1].plot(self.before[:,0], yb, 'r-')
        ax[1].plot(self.before[:,0], yb1, 'g--')
        ax[1].plot(self.before[:,0], yb2, 'g--')
        ax[1].fill_between(self.before[:,0], yb1, yb2, facecolor="gray", alpha=0.15)

        ax[1].plot(self.after[:,0], ya, 'b-')
        ax[1].plot(self.after[:,0], ya1, 'm--')
        ax[1].plot(self.after[:,0], ya2, 'm--')
        ax[1].fill_between(self.after[:,0], ya1, ya2, facecolor="gray", alpha=0.15)

        
        ax[1].set_xlabel("Surface Depth")
        ax[1].set_ylabel("Alpha-beta fraction")        
        ax[1].set_title('Alpha-beta volume fraction with surface depth')
        fig.tight_layout()
        plt.show()
        
        

        
    
    
class PreprocessContainer:
    def __init__(self, analysis_method: Type[DataAnalysis], image_directory: str, data_directory: str, denudation_data:str):

        self.denudation_images  = np.loadtxt(denudation_data, usecols=(0,), skiprows=1)

        self.image_directory = image_directory
        self.images = self.sort_files_by_data(os.listdir(self.image_directory), self.denudation_images)

        self.data_directory = data_directory
        self.data_files = self.sort_analysis_files_by_data(os.listdir(self.data_directory), self.denudation_images)
        
        self.method = analysis_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")

    def sort_files_by_data(self, files, ordered_list, condition_func=None):
        names = []
        for o in ordered_list:
            for f in files:
                if f.startswith(o):
                    names.append(f)
        print(len(names), len(ordered_list), len(files))
        return names

    def sort_analysis_files_by_data(self, files, ordered_list, condition_func=None):
        names = []
        for o in ordered_list:
            for f in files:
                base_name = os.path.splitext(f.split('_')[1])[0]
                if base_name == o:
                    names.append(f)
        print(len(names), len(ordered_list), len(files))
        return names


    def get_filename_from_id(self, name):
        analysis_name = os.path.splitext(name.split('_')[0])[0]
        base_name = os.path.splitext(name.split('_')[1])[0]
        extension = os.path.splitext(name)[1]
        prefixed = [filename for filename in os.listdir(self.image_directory) if filename.startswith(base_name)]
        
        print("Unprocessed image: ", prefixed[0])
        return prefixed[0]

    def analyse_images(self, plot=True):
        for data_file in self.data_files:
            unprocessed_name = self.get_filename_from_id(image)
            unprocessed_image =   color.rgb2gray(imageio.imread(f"{self.original_image_directory}/{unprocessed_name}"))

            # This gives the pixel depth, alpha beta volume fraction data
            data = np.loadtxt(f"{self.data_directory}/{data_file}", usecols=(0,1), skiprows=1)

            self.method.analyse( data )
            if plot:
                self.method.plot(original_image)
            
            self.save(image, self.method.data)


    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.dt}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name
    

    def save(self, image_name, data, mode='w'):
        self.dir_name = self.create_output_directory()
        name = os.path.splitext(image_name)[0]
        path = os.path.join(self.dir_name, f"{self.method_name}_{name}.dat")
        
        
        print(f"> Saving data to {path}")
        with open(path, mode) as f:

            if isinstance(self.method.data, np.ndarray):
            
                f.write(self.method.comment + "\n")

                for row in self.method.data:
                    data_line = (' '.join(["{: 12.6f}" * len(row)])).format( *tuple(row) ) + "\n"
                    f.write(data_line)
            elif isinstance(self.method.data, str):
                f.write(self.method.data)
        
    
