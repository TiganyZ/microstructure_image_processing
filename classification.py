#!/usr/bin/python3

"""This file is used to test the hypotheses to classify a given set of
data extracted from an image.

It will classify the data to see if its corresponding image exibits beta denudation,
has an even distribution of alpha and beta, or that we see the
opposite trend.

We can use a naive bayes classifier as a first stab, to classify the
boundary between a beta denudated reduction from the surface, and the
actual bulk. From this we can draw a line across the image, which
informs us where the bulk volume fraction is found with a given
confidence.

A prior distribution, which encodes how likely we think there could be
beta denudation in a sample could be encoded from a global gradient.

We will use Bayes rule to compute this.

Assumptions made about the volume fraction data: 

- In the bulk, there is a gaussian distribution of volume fractions
  around the mean 
- In the surface region, there is a gaussian
  distribution about a mean which decreases with depth


We have a hierarchy of classifications: 
* We must classify if the distribution of points corresponds to one of three hypothesis. 
   - We have a prior which is biased in favour of denudation rather than no denudation/anti-denudation
   - This specifies that we need to choose either between two classes for the data or one:
     1. One class applies if one sees that the distribution of volume fractions is flat
     2. Two classes apply if one sees that the distribution of volume fractions is not flat in given parts
        - This applies to both the dendudation and anti-denudation hypotheses. 

* To classify the data, one can either take it how it is, and see if a
  given straight line can fit it, or

* One can assume a gaussian which is distributed about the volume
  fraction, (y-axis), therefore for each point, we can see if it resides within a given gaussian. 

* We do not know what the bulk volume fraction is. How do we measure
  this since the true bulk volume fraction depends on the hypothesis
  chosen.
* Is taking the data in its totality fair to make an assessment of the bulk volume fraction?

* Really, this is a regression task, as the mean is a continuous
  variable, and one can try to distinguish between the two models by
  the variation of this parameter


*** Testing between models/hypotheses for data ***
- One can do a LDA to see if given points are within a given mean or not. 
- The depth information will inform us also 

One can do PCA (Principal Component Analysis) such that one can find the eigenvectors which maximise the variance of the class data. 


* Instead of making things complicated, I can do something else: 
- For the testing of denudation, I could split the data in half, and then fit two lines in the data. 
- I can obtain an r value for each 
- Then I can change the dividing line and do the same again. 


* Classify can return a confidence of the fitting
* The decision boundary should have the prior imposed, if we 


We split the data set into testing and training. 
One can do this multiple times, and perform statistics on a given dataset 


"""

# Make an abstract base class which supplies a processing function upon an image
class DataClassification(ABC):
    
    @abstractmethod
    def classify(self, data):
        pass
            
    @abstractmethod
    def plot(self, data):
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
    

class BayesianHypothesis(DataClassification):
    def __init__(self, data):
        self.fit = LinearFit()
        self.data = data
    
    def compare_methods(self, linear_data, flat_data):
        params_flat,     stdevs_flat     = self.fit(**self.flat(flat_data))
        params_straight, stdevs_straight = self.fit(**self.straight(linear_data))

        # compare the standard deviation between the points and see what fits better
        std_fint = stdevs[0]
        (std_lgrad, std_lint) = stdevs_straight[0], stdevs_straight[1]
        
        
    def priors(self):
        d = {
            "straight" : 0.5,
            "flat" : 0.4,
            "neither" : 0.1
             }
        return d


    # Possibly bootstrap the samples for the averages, 
    
    def posterior(self, prior, likelihood, evidence=1.0):
        return likelihood * prior / evidence
    
    def bayesian_hypothesis_test(self):
        prior = self.priors()

        # Here we test the hypotheses, based on the data we botain
        # Give initial line
        line = (np.max(self.data[:,0]) - np.min(self.data[:,0])) / 2.
        
        segment = SegmentData(self.data, line)
        before, after = segment.segment_data()

        # What I can do actually is first split the data into train
        # and test sets and then to the fitting, then quantify an error on each of the fits.

        # This error for the fitting will be used to test the hypotheses

        
        
class BayesClassifier(DataClassification):
    


class ClassificationContainer:
    def __init__(self, classification_method: Type[DataClassification], data_directory: str, image_directory: str):
        self.data_directory = data_directory
        self.image_directory = image_directory        
        self.data = os.listdir(self.data_directory)

        self.method = classification_method()
        self.method_name = self.method.name

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")

    def get_filename_from_id(self, name):
        
        analysis_name = os.path.splitext(name.split('_')[0])[0]
        base_name = os.path.splitext(name.split('_')[1])[0]
        extension = os.path.splitext(name)[1]
        prefixed = [filename for filename in os.listdir(self.image_directory) if filename.startswith(base_name)]

        # Assuming just one identifier from image name, can't deal with multiple images'
        print("Unprocessed image: ", prefixed[0])
        return prefixed[0]

    def classify_images(self, plot=True):

        for data in self.data_directory:
            unprocessed_name = self.get_filename_from_id(image)
            original_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{unprocessed_name}"))
            self.method.classify( original_image, unprocessed_image )
            if plot:
                self.method.plot(original_image)
            
            self.save(image, self.method.data)
            

            
        for image in self.images:
            original_image =   color.rgb2gray(imageio.imread(f"{self.image_directory}/{image}"))
            unprocessed_name = self.get_filename_from_id(image)
            unprocessed_image =   color.rgb2gray(imageio.imread(f"{self.original_image_directory}/{unprocessed_name}"))
            self.method.analyse( original_image, unprocessed_image )
            if plot:
                self.method.plot(original_image)
            
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

    plot=False
    image_directory = "images_RemoveBakelite_WhiteBackgroundRemoval_OtsuThreshold"
    
    print(f"Analysing images from {image_directory}..")    
    # Now analyse
    print(f"> Alpha-beta fraction")    
    analysis = AlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory)
    ac.analyse_images(plot=plot)

    print(f"> Chris Alpha-beta fraction")    
    analysis = ChrisAlphaBetaFraction
    ac = AnalysisContainer(analysis, image_directory)
    ac.analyse_images(plot=plot)

    
    
    
        



