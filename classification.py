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


A method we can do is for each set of data:
- Segment data into training, test and validation sets
- Try a threshold to classify two data sets
- Perform logistic regression or a t-test to see if the difference in the means is significant
  - Logistic regression gives a likelihood for a given classification
  - We can optimise the logistic regression to get the threshold which corresponds to the maximum likelihood
- If the means are significant, fit lines to the data
  - Fit a linear regression to the first set, and a flat distribution to the second

- Can look at type of R squared relationship (SS(mean)-SS(fit)) /
  SS(mean), to see how well the fit is compared to the mean. (how good
  of a predictor)

- We can see if the R squared value of a mean over the whole data is better than the segmentation over the whole
- This will always be in favour of the two models as there are two regressions which minimise each of squares.

- For logistic regression, once we have the fit, and the quantities
  for the pseudo-R-squared, then we can find a p value
  (2(LL(fit)-LL(mean))) which corresponds to a point on a Chi-squared distribution

- But, one can compare the mean of the data can


* Bayesian statistics

Using a Bayesian approach, one gives distributions to each of the unknown quantities. 

In our case, the quantity we want to determine is the threshold, given there is beta denudation. 

We can perform bayesian analysis in multiple ways: 

1. Hypothesis testing
   - Our prior is the hypothesis that there is:
     > Beta denudation
     > Not beta denudation - flat distribution 
     > Neither
   - These are distributions which are of the mean value of the alpha volume fraction in a given region
   - If we expect beta denudation, the prior would maybe be a gaussian which is centred quite high in terms of volume fraction
   - We can assume a flat prior: for whatever mean value of alpha, we
     expect beta denudation / not beta denudation to be equally likely
      



1. We can associate a posterior distribution which encodes
   - Our prior belief that there is beta denudation in the samples tested
     > We may assume a uniform distribution, that beta denudation is equally likely



What we can do is find the threshold by using ROC AUC scores. 

    1. Fit Model on the Training Dataset.
    2. Predict Probabilities on the Test Dataset.
    3. For each threshold in Thresholds:
        3a. Convert probabilities to Class Labels using the threshold.
        3b. Evaluate Class Labels.
        3c. If Score is Better than Best Score.
            3ci. Adopt Threshold.
        4. Use Adopted Threshold When Making Class Predictions on New Data.


To actually pick a threshold, one can use the training data, to see if
such a threshold exists. Prior to this, it must be possible to see a
trend in the data.

There will be a distribution of the thresholds that we can choose, we
can choose the one which minimises the error on the fit statistics. 

One must first be able to determine if there is really beta denudation in the training data

As such we can do hypothesis testing which is available to use through a bayesian framework. 

We can test the general trend of the data and ask scientific questions about this trend: 

- We can see the gradient of this trend which we obtain by fitting the test data
  > With the bayesian method, we can see there will be a distribution of potential gradients which one could obtain 
  > This distribution is based on the prior distribution we associate to it, which will be flat
  > We can test to see if a given gradient value, or greater, is 


We want

P(Denudation | gradient) = P(gradient | denudation) * P(denudation) / P(gradient)


Remember that the first parameters, are the ones which are being varied. 

So we have a given prior which is P(denudation), which gives a distribution over the possible gradients of what we expect is denudated. 

We can expect that 
P(denudation) = 0 if gradient is > 0
P(denudation) = exponential distribution if gradient < 0
              = Some logistic curve which is is centred around some negative gradient

The data likelihood, P(gradient | denudation) is the distribution/likelihood of gradients which we have got from the data given that there is denudation. 

We can introduce a denudation parameter, which encodes the information
that there is maximal denudation when the gradient is negative
infinity, and 0 when the gradient is zero.

To get the likelihood, we first assume a statistical model.

We can assume that the gradients drawn are from a gaussian probability distribution, from the data we can obtain 
We can use a logistic curve as the probability that there is denudation


As such we can assume the likelihood function, which gives the likelihood of a gradient, given an amount of denudation.

One can actually get a denudation distribution given the gradients which are found. This is what we would like to determine. 



The likelihood function can be generated from the points sampled, we must assume a distribution. 
Say the points are sampled from a distribution, then we can say that 



Given denudation, we expect a gaussian distribution of gradients,
which we can obtain in the usual way: a gaussian which is defined from
the fit statistics from the testing data. 

We need to construct the likelihood function. We can do this is a similar way as for a coin-flip test. 

We can assume that the target variable is distributed with gaussian noise, which is logical. 
P(t | X, w^T.\phi(x), beta^-1)

P(gradient | denudation ) is a likelihood function which states that,
given that the amount of denudation can vary, say a denudation parameter, how likely is a given gradient? 


Assuming that we can pick the gradient from a normal distribution, this is the observation yi from the data
We measure a gradient, which has a particular error associated with it

The likelihood will take 

We can assume that the 

If we vary the gradient, what is the likelihood of the denudation?

We expect the gradients sampled to be from a normal distribution. 

This means, for a given sample, assuming they are independent and
identically distributed, the likelihood function will be the joint
probability distribution from a sampled gradient, which will be a product of gaussians. 

We can actually obtain from all of the data, a prediction of denudation given gradient information from samples of the alpha beta fraction.

One could create an adversarial network to do this. 

--------------------------------------------------
--------------- Types of modelling ---------------
--------------------------------------------------

--------------------
Supervised learning
--------------------

- Given the dataset, we can train the predictive model on a subset of the data. 

- This model can then predict, given data, which we fit a line to,
  which has a particular gradient, how likely it is that there is beta denudation.

- This would necessitate human input of which images are denudated and which aren't. 

--------------------------------------------------

--------------------
Denudation detection
--------------------

- To classify if there is denudation, we can have two hypothesis with given probabilities
1. P(Denudation    | data) = P(data |    Denudation) * P(   Denudation) /P(data)  =>  P(H0|Y=y) = f(y|H0).P(H0)/P(y)
2. P(No Denudation | data) = P(data | No Denudation) * P(No Denudation) /P(data)  =>  P(H1|Y=y) = f(y|H1).P(H1)/P(y)

Say Denudation => Y=1
 No Denudation => Y=0

1. P(Y=1 | data) = P(data |Y=0) * P(Y=1)/P(data)
2. P(Y=0 | data) = P(data |Y=1) * P(Y=0)/P(data)

____ Do the above with logistic function as the likelihood? ____

- Problem is, we need to know the actual supervised vaues to train the
  logistic regression on, such that we can even do the hypothesis
  testing. Therefore, we can classify if each of the images are
  denudated or not, by human eye, then use this to train the logistic
  regression. Then, we can test the model on the rest of the data and see the predictions.


- Now we know the distribution of the data that we expect, under the two different hypotheses
  > We would expect that there is a gradient in the fit of the data which is negative
  > What is the likelihood function / the distribution that we assume?
  > We can assume that the denudation a function which
    - If the gradient is less than 0, then there is a very small chance that there is 
  > This actually logistic regression if we do this! As we assume that the likelihood is a logistic function: 
    - The likelihood function we have is:
    - L(theta) = \P_i=1..N P(Y=1|x=xi ; theta) * \P_i=1..N (1 - P(Y=1|x=xi ; theta) )

- We assume the model that P(y=1|x=xi) = \sigma(theta_0 + theta_1 * x)
  > Where \sigma(z) = 1 / (1 + e^{-z})
  > Now we can find the likelihood function which will be 

Lets now find the likelihood function and then minmise by minimising the negative log likelihood

-log( L(theta) ) =  −∑i=1...N  yi * log(P(Yi=1|X=x;Θ)) + (1−yi) log(P(Yi=0|X=x;Θ))


 
> This means we know the likelihood function/distribution of P(data | H)


--- Denudation Hypothesis ---
- Assume that the distribution is 

How likely is the data we have, given that there is denudation. 
We need to have a model for denudation such that we can estimate the likelihood. 

- If hypothesis 1 is greater than 2, then we have denudation. 
- 

Globally, given gradients from all of the data, one can define this: 
1. P(data |    Denudation) = P(Denudation    | data) * P(data)
2. P(data | No Denudation) = P(No Denudation | data) * P(data)

Which are the gradients found form each sample. 
1. P(g1, g2, g3,..., gn; threshold |    Denudation) = P(Denudation    | g1, g2, g3,..., gn; threshold) * P(g1, g2, g3,..., gn; threshold)
2. P(g1, g2, g3,..., gn; threshold | No Denudation) = P(No Denudation | g1, g2, g3,..., gn; threshold) * P(g1, g2, g3,..., gn; threshold)


- Here, we know that the likelihood function will be a joint distribution 
- We can assume different distributions, say 
  > Binomial distribution which is if there is denudation observed or not
  > 
- We vary the denudation parameter, to see how the likelihood of the data changes
- We know that

"""
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, f1_score
from matplotlib import pyplot



# Make an abstract base class which supplies a processing function upon an image
class DataClassification(ABC):
    
    @abstractmethod
    def classify(self, data):
        pass
            
    @abstractmethod
    def plot(self, data):
        pass


class ThresholdDetermination:

    def brute_force(self, x, y):
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
        model = LogisticRegression(solver='lbfgs')
        model.fit(trainX, trainy)
        # predict probabilities
        yhat = model.predict_proba(testX)
        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]

        # define thresholds
        thresholds = arange(0, 1, 0.001)
        # evaluate each threshold
        scores = [f1_score(testy, to_labels(probs, t)) for t in thresholds]
        # get best threshold
        ix = argmax(scores)
        print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
        best_thresh = thresholds[ix]
        print('Best Threshold=%f' % (best_thresh))
        return best_thresh

    def plot(self):
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(self.fpr, self.tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()

class ROCThresholdDetermination:    
    def roc_auc_analysis(self, x, y):
        # roc curve for logistic regression model
        # generate dataset
        # X, y = make_classification(n_samples=10000, n_features=2, n_redundant=0,
        #         n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=4)
        # split into train/test sets
        trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, random_state=2, stratify=y)
        # fit a model
        model = LogisticRegression(solver='lbfgs')
        model.fit(trainX, trainy)
        # predict probabilities
        yhat = model.predict_proba(testX)
        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]
        # calculate roc curves
        self.fpr, self.tpr, thresholds = roc_curve(testy, yhat)
    
        # get the best threshold using J statistic 
        J = self.tpr - self.fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        print('Best Threshold=%f' % (best_thresh))
        return best_thresh

    def plot(self):
        # plot the roc curve for the model
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        plt.plot(self.fpr, self.tpr, marker='.', label='Logistic')
        # axis labels
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        # show the plot
        plt.show()
    
        
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
    

class BayesianLogisticHypothesis(DataClassification):
    def __init__(self, data):
        self.fit = LinearFit()
        self.data = data

    # Now train the model on the data

    
    
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

    
    
    
        



