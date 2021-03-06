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

-log( L(theta) ) =  ??????i=1...N  yi * log(P(Yi=1|X=x;??)) + (1???yi) log(P(Yi=0|X=x;??))


 
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



----------------------------------------
           Testing the data
----------------------------------------
Models to test:
- Logistic regression
- Support vector machines 

Procedure:
 1) Get gradient data for the dataset as a first measure: just a simple straight line per dataset. 
   - For a more involved approach, we can fit a curve to the data using a linear regression with basis functions. 
 2) Split the full dataset (the human-classified denudation data) into train and test sets
 3) Use k-fold cross validation on the models to get scores 
    - Best threshold from the ROC curve can be done for each curve
 5) Test the model on the test dataset to provide an arror. 
 6) Interpret the models, in terms of bayes theorem, for what it means. 


"""
from abc import ABC, abstractmethod
import os
from typing import Type, TypeVar
import numpy as np

from datetime import datetime

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import roc_curve, f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import preprocessing
import matplotlib.cm as cm

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess_data import LinearFit
from scipy.optimize import curve_fit, minimize
# Make an abstract base class which supplies a processing function upon an image
class DataTrainer(ABC):
    
    @abstractmethod
    def train(self, data):
        pass
            
    @abstractmethod
    def plot(self, data):
        pass

 

class DataClassification(ABC):
    
    @abstractmethod
    def classify(self, data):
        pass
            
    @abstractmethod
    def plot(self, data):
        pass

    
# Make an abstract base class which supplies a processing function upon an image
class DataAnalysis(ABC):
    
    @abstractmethod
    def analyse(self, image):
        pass
            
    @abstractmethod
    def plot(self, image):
        pass

        
class ROCThresholdDetermination(DataAnalysis):
    def __init__(self, X, y, model=LogisticRegression(solver='lbfgs'), test_size = 0.3):
        self.X = X
        self.y = y
        self.model = model
        self.test_size = test_size
        

    
    def analysis(self):
        # roc curve for logistic regression model
        # split into train/test sets
        
        #trainX, testX, trainy, testy = train_test_split(self.X, self.y, test_size=self.test_size, random_state=2, stratify=y)
        # fit a model
        model.fit(trainX, trainy)
        yhat = model.predict_proba(testX)

        # keep probabilities for the positive outcome only
        yhat = yhat[:, 1]
        self.fpr, self.tpr, thresholds = roc_curve(testy, yhat)
    
        # get the best threshold using J statistic 
        J = self.tpr - self.fpr
        ix = argmax(J)
        best_thresh = thresholds[ix]
        print('Best Threshold=%f' % (best_thresh))
        self.threshold = self.best_thresh
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
    

class BayesianLogisticHypothesis(DataClassification):
    def __init__(self, data):
        self.fit = LinearFit()
        self.data = data

    # Now train the model on the data
    def priors(self):
        d = {
            "straight" : 0.5,
            "flat" : 0.4,
            "neither" : 0.1
             }
        return d

    def posterior(self, prior, likelihood, evidence=1.0):
        return likelihood * prior / evidence


    
class TrainSVM(DataTrainer):
    def __init__(self):
        self.name = "Support Vector Classifier"

    def train(self, X, y):

        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=5)
        # define the model
        model = svm.SVC(kernel='linear', random_state=42,probability=True )
        # define search space
        space = dict()
        linear=True

        if not linear:
            space['C'] = [0.01, 0.1]
            space['kernel'] = ['rbf']
            space['gamma'] = [0.1, 0.5, 1, 2 ]
        else:
            space['C'] = [0.01, 0.1, 1, 100]
            #            space['kernel'] = ['linear']
        #        space['gamma'] = [0.1, 0.5, 1, 2 ]        
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X, y)
        # get the best performing model fit on the whole training set
        self.model = result.best_estimator_
        
        # skf = StratifiedKFold(n_splits=3)
        # for train, test in skf.split(X, y):
        
        # self.model = svm.SVC(kernel='linear', C=1, random_state=42,probability=True ).fit(X,y)
        # scores = cross_val_score(self.model, X, y, cv=5)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        return self.model#, scores.mean()
    
    def plot(self, X_test, X_original, y, X_train_transformed, y_train, scaler):
        probs = self.model.predict_proba(X_test)
        # Plot the test data, the logistic regression curve and the ROC

        fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

        X_train = scaler.inverse_transform(X_train_transformed)
        
        ax[0].set_title("SVM training")
        ax[0].scatter(X_train[:,0][y_train == 0], X_train[:,1][y_train == 0],  alpha=0.2, color='blue', label="Training data--normal")        
        ax[0].scatter(X_train[:,0][y_train == 1], X_train[:,1][y_train == 1],  alpha=0.2, color='red', label="Training data--denudated")

        ax[0].scatter(X_original[:,0][y == 0], X_original[:,1][y == 0], color='purple', alpha=0.5, label="Testing data--normal")
        ax[0].scatter(X_original[:,0][y == 1], X_original[:,1][y == 1], color='yellow', alpha=0.5, label="Testing data--denudated")
                      
        ax[0].set_xlabel("Surface-to-bulk ratio")
        ax[0].set_ylabel("Classification space")            
        ax[0].legend()
        ax[0].legend(loc='upper right',
                     ncol=1, borderaxespad=0.)
        
        leg = ax[0].get_legend()
        # leg.legendHandles[0].set_color('red')
        # leg.legendHandles[1].set_color('blue')        
        # leg.legendHandles[2].set_color('yellow')
        # leg.legendHandles[3].set_color('purple')        
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()

        # create grid to evaluate model
        # xlim = (min(np.min(X_train[:,0]), np.min(X_test[:,0])), max(np.max(X_train[:,0]), np.max(X_test[:,0])))
        # print(xlim)
        # ymax = max(np.max(y_train), np.max(y))
        # ymin = min(np.min(y_train), np.min(y))        

        xx = np.linspace(xlim[0], xlim[1], 30)
        scaler = preprocessing.StandardScaler().fit(xx.reshape(xx.shape + (1,)))
        xxt = scaler.transform(xx.reshape(xx.shape + (1,)))[:,0]
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xxt)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.model.decision_function(xy).reshape(XX.shape)

        YY, XX = np.meshgrid(yy, xx)
        # ax[0].imshow(
        #     Z,
        #     interpolation="nearest",
        #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        #     aspect="auto",
        #     origin="lower",
        #     cmap=plt.cm.magma,
        # )
        # plot decision boundary and margins
        ax[0].contour(
            XX, YY, Z, colors="k", levels=[-0.5, 0, 0.5], alpha=0.5, linestyles=["--", "-", "--"]
        )
        # plot support vectors
        # ax[0].scatter(
        #     self.model.support_vectors_[:, 0],
        #     self.model.support_vectors_[:, 1],
        #     s=10,
        #     linewidth=1,
        #     facecolors="none",
        #     edgecolors="k",
        # )
        ordered = np.argsort(X_original[:,0])
        ax[1].plot(X_original[:,0][ordered], probs[:,1][ordered], label="Denudation risk")#
        ax[1].plot(X_original[:,0][ordered], probs[:,0][ordered], alpha=0.5, label="Normal risk")#
        ax[1].set_title(f'SVM predicted probability')
        ax[1].set_ylabel("Risk")
        ax[1].set_xlabel("Surface-to-bulk ratio")
        ax[1].legend()

        # ax[1].plot(X_original[:,0], probs[:,0], 'bo', label="Testing data")#
        # ax[1].set_title(f'SVM predicted probability')
        # ax[1].set_ylabel("Risk")
        # ax[1].set_xlabel("Surface-to-bulk ratio")        

        # ax[2].plot(X_original[:,1], probs[:,1], 'ro')        
        # ax[2].set_title(f'SVC predicted probability')
        # ax[2].set_ylabel("Risk")

        
        fig.tight_layout()
        plt.show()
        

  
class TrainLogisticRegression(DataTrainer):
    def __init__(self):
        self.name = "Logistic Regression"
    
    def train(self, X, y):
        # logreg=LogisticRegression()

        # predicted = cross_validation.cross_val_predict(logreg, X, y, cv=10)
        # print("Logistic regression: \n", metrics.accuracy_score(y, predicted) )
        # print("Logistic regression: \n", metrics.classification_report(y, predicted))

        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=3)
        # define the model
        model = LogisticRegression() #(class_weight = 'balanced') #svm.SVC(kernel='linear', random_state=42,probability=True )
        # define search space
        space = dict()
        space['C'] = [0.01, 0.1, 1, 10, 1000]
        # define search
        search = GridSearchCV(model, space, scoring='accuracy', cv=cv_inner, refit=True)
        # execute search
        result = search.fit(X, y)
        # get the best performing model fit on the whole training set
        self.model = result.best_estimator_



        # # calculate roc curves
        # fpr, tpr, thresholds = roc_curve(testy, yhat)
        # # get the best threshold
        # J = tpr - fpr
        # ix = argmax(J)
        # best_thresh = thresholds[ix]
        # print('Best Threshold=%f' % (best_thresh))
        
        # logreg=LogisticRegression()
        # self.model = logreg.fit(X, y)
        # scores = cross_val_score(self.model, X, y, cv=5)
        # print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
        # threshold = 0.5
        # log_reg.predict_proba(X_test)
        return self.model#, scores.mean()

    def plot(self, X_test, X_original, y, X_train_transformed, y_train, scaler):

        probs = self.model.predict_proba(X_test)
        # Plot the test data, the logistic regression curve and the ROC

        fig, ax = plt.subplots(ncols=2, figsize=(8, 3))

        X_train = scaler.inverse_transform(X_train_transformed)
        
        ax[0].set_title("LogReg training")
        ax[0].scatter(X_train[:,0][y_train == 0], X_train[:,1][y_train == 0],  alpha=0.2, color='blue', label="Training data--normal")        
        ax[0].scatter(X_train[:,0][y_train == 1], X_train[:,1][y_train == 1],  alpha=0.2, color='red', label="Training data--denudated")

        ax[0].scatter(X_original[:,0][y == 0], X_original[:,1][y == 0], color='purple', alpha=0.5, label="Testing data--normal")
        ax[0].scatter(X_original[:,0][y == 1], X_original[:,1][y == 1], color='yellow', alpha=0.5, label="Testing data--denudated")
                      
        ax[0].set_xlabel("Surface-to-bulk ratio")
        ax[0].set_ylabel("Classification space")            
        ax[0].legend()
        leg = ax[0].get_legend()
        # leg.legendHandles[0].set_color('red')
        # leg.legendHandles[1].set_color('blue')        
        # leg.legendHandles[2].set_color('yellow')
        # leg.legendHandles[3].set_color('purple')        
        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()

        # create grid to evaluate model
        # xlim = (min(np.min(X_train[:,0]), np.min(X_test[:,0])), max(np.max(X_train[:,0]), np.max(X_test[:,0])))
        # print(xlim)
        # ymax = max(np.max(y_train), np.max(y))
        # ymin = min(np.min(y_train), np.min(y))        

        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)

        scaler = preprocessing.StandardScaler().fit(xx.reshape(xx.shape + (1,)))
        xxt = scaler.transform(xx.reshape(xx.shape + (1,)))[:,0]

        scaler = preprocessing.StandardScaler().fit(yy.reshape(yy.shape + (1,)))
        yyt = scaler.transform(yy.reshape(yy.shape + (1,)))[:,0]
        
        
        # yyt = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yyt, xxt)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.model.decision_function(xy).reshape(XX.shape)

        YY, XX = np.meshgrid(yy, xx)
        # ax[0].imshow(
        #     Z,
        #     interpolation="nearest",
        #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        #     aspect="auto",
        #     origin="lower",
        #     cmap=plt.cm.magma,
        # )
        # plot decision boundary and margins
        ax[0].contour(
            XX, YY, Z, colors="k", levels=[-0.5, 0, 0.5], alpha=0.5, linestyles=["--", "-", "--"]
        )
        # plot support vectors
        # ax[0].scatter(
        #     self.model.support_vectors_[:, 0],
        #     self.model.support_vectors_[:, 1],
        #     s=10,
        #     linewidth=1,
        #     facecolors="none",
        #     edgecolors="k",
        # )

        ordered = np.argsort(X_original[:,0])
        # ax[1].plot(X_original[:,0][ordered], probs[:,0][ordered], label="")#

        ax[1].plot(X_original[:,0][ordered], probs[:,1][ordered], label="Denudation risk")#
        ax[1].plot(X_original[:,0][ordered], probs[:,0][ordered], alpha=0.5, label="Normal risk")#

        ax[1].set_title(f'LogReg predicted probability')
        ax[1].set_ylabel("Risk")
        ax[1].set_xlabel("Surface-to-bulk ratio")
        ax[1].legend()

        # ax[2].plot(X_original[:,1], probs[:,1], 'ro')        
        # ax[2].set_title(f'SVC predicted probability')
        # ax[2].set_ylabel("Risk")

        
        fig.tight_layout()
        plt.show()
        

        probs = self.model.predict_proba(X_test)
        
  
        fig, ax = plt.subplots(ncols=3, figsize=(12, 3))

        ax[0].scatter(X_train[:,0], X_train[:,1], c=y_train, alpha=0.5,  cmap='Spectral')
        ax[0].scatter(X_test[:,0], X_test[:,1], c=y, alpha=0.5)
        ax[0].set_xlabel("Surface gradient")
        ax[0].set_ylabel("Surface to bulk ratio")            


        xlim = ax[0].get_xlim()
        ylim = ax[0].get_ylim()

        # create grid to evaluate model
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.model.decision_function(xy).reshape(XX.shape)


        # ax[0].imshow(
        #     Z,
        #     interpolation="nearest",
        #     extent=(xx.min(), xx.max(), yy.min(), yy.max()),
        #     aspect="auto",
        #     origin="lower",
        #     cmap=plt.cm.magma,
        # )
        # plot decision boundary and margins
        ax[0].contour(
            XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
        )
        # plot support vectors
        # ax[0].scatter(
        #     self.model.support_vectors_[:, 0],
        #     self.model.support_vectors_[:, 1],
        #     s=10,
        #     linewidth=1,
        #     facecolors="none",
        #     edgecolors="k",
        # )

        ax[1].plot(X_original[:,0], probs[:,0], 'bo')#
        ax[1].set_title(f'LogReg predicted probability')
        ax[1].set_ylabel("Risk")

        ax[2].plot(X_original[:,1], probs[:,1], 'ro')        
        ax[2].set_title(f'LogReg predicted probability')
        ax[2].set_ylabel("Risk")

        
        fig.tight_layout()
        plt.show()
        
        
        fig, ax = plt.subplots(nrows=len(X_test.T), ncols=2, figsize=(8, 3*len(X_test.T)))

        #        X_original = scaler.inverse_transform(X_test)

        probs = self.model.predict_proba(X_test)
        importance = self.model.coef_[0]

        # # summarize feature importance
        # for i,v in enumerate(importance):
	#     print('Feature: %0d, Score: %.5f' % (i,v))
        #     # plot feature importance
        #     pyplot.bar([x for x in range(len(importance))], importance)
        #     pyplot.show()        
        for i, (X,pred) in enumerate(zip( X_original.T, probs.T)):
            
            ax[i, 0].set_title(f'X_test {i}')
            ax[i, 0].plot(X, y, 'b+')
            ax[i, 0].set_ylabel("Classification")

            #             xlim = ax[i,0].get_xlim()
            # ylim = ax[i,0].get_ylim()

            # # create grid to evaluate model
            # xx = np.linspace(xlim[0], xlim[1], 30)
            # yy = np.linspace(ylim[0], ylim[1], 30)
            # YY, XX = np.meshgrid(yy, xx)
            # xy = np.vstack([XX.ravel(), YY.ravel()]).T
            # Z = model.decision_function(xy).reshape(XX.shape)


            # # plot decision boundary and margins
            # ax[i,0].contour(
            #     XX, YY, Z, colors="k", levels=[-1, 0, 1], alpha=0.5, linestyles=["--", "-", "--"]
            # )

            ax[i, 1].plot(X, pred, 'ro')
            ax[i, 1].set_title(f'{self.name} predicted probability')
            ax[i, 1].set_ylabel("Risk")
        fig.tight_layout()
        plt.show()
        

    

    
class ClassificationContainer:
    def __init__(self, classification_methods, data: str, image_directory: str, classification_data: str, classify_from_file=True, optimise_threshold=False):
        self.image_directory = image_directory        
        self.images = np.loadtxt(data, skiprows=1, usecols=(0,), dtype=np.str)
        self.optimise_threshold = optimise_threshold

        # Assumed one has images in first column and surface to bulk ratios in next
        self.data_to_test = np.loadtxt(data, skiprows=1, usecols=(1,), dtype=np.float)
        
        self.X = np.zeros((len(self.data_to_test), 2))
        self.y = np.random.rand((len(self.data_to_test)))
        
        if classify_from_file:
            print(f"\n >> Training model with data from {classification_data}")
            classification_images = np.loadtxt(classification_data, skiprows=1, usecols=(0,), dtype=np.str)
            ratios = np.loadtxt(classification_data, skiprows=1, usecols=(1,), dtype=np.float)
            classification = np.loadtxt(classification_data, skiprows=1, usecols=(2,), dtype=np.int)

            self.X = np.zeros((len(classification_images), 2))
            self.y = classification
            
            self.X[:,0] = ratios

        else:
            # Training from data itself
            self.X[:,1] = self.data_to_test


        self.X[:,1] = 0.

        # split into train test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=56, stratify=self.y)

        # self.X_train = self.X
        # self.y_train = self.y
        # self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=56, stratify=self.y)
        self.scaler = preprocessing.StandardScaler().fit(self.X_train)
        self.X_train_transformed = self.scaler.transform(self.X_train)
        self.X_test_transformed  = self.scaler.transform(self.X_test)
        
        self.methods = classification_methods

        now = datetime.now()
        self.dt = now.strftime("%Y-%m-%d--%H-%M-%S")


    def train_models(self, plot=True):

        for method in self.methods:
            trainer = method()
            model = trainer.train(self.X_train_transformed, self.y_train)

            yhat = model.predict(self.X_test_transformed)
            if self.optimise_threshold:
                yhat = model.predict_proba(self.X_test_transformed)
                yhat = yhat[:, 1]
                # calculate pr-curve
                precision, recall, thresholds = precision_recall_curve(self.y_test, yhat)
                fscore = (2 * precision * recall) / (precision + recall)
                # locate the index of the largest f score
                ix = np.argmax(fscore)
                print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
                best_thresh = thresholds[ix]
                # plot the roc curve for thebest_thresh = thresholds[ix model
                no_skill = len(self.y_test[self.y_test==1]) / len(self.y_test)
                plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
                plt.plot(recall, precision, marker='.', label='Logistic')
                # axis labels
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.legend()
                # show the plot
                plt.show()

                # yhat = np.zeros(yhat.shape, dtype=bool)
                yhat = model.predict_proba(self.X_test_transformed)[:,1] > best_thresh            
                # evaluate the model
            acc = accuracy_score(self.y_test, yhat)

            data_to_transform = np.zeros((len(self.data_to_test), 2))
            data_to_transform[:,0] = self.data_to_test
            data_transformed = self.scaler.transform(data_to_transform)

            if self.optimise_threshold:
                yhat = model.predict_proba(data_to_transform)[:,1] > best_thresh 
            else:
                yhat = model.predict(self.X_test_transformed)
            
            print(trainer.name, "\n", " > Score = ", acc)
            for yp, yt in zip(yhat, self.y_test):
                print(" y: ", yp, yt, yp == yt)
            
            self.save_classification_results(f"results_classifications_{self.dt}.txt", self.data_to_test, yhat)

            if plot:
                trainer.plot(self.X_test_transformed, self.X_test, self.y_test, self.X_train_transformed, self.y_train, self.scaler)
            
                
    def polyfit(self, x, y, degree):
        results = {}
    
        def straight_line(x, m, c):
            return m*x + c

        expected = (1., 0.0)
        params,cov=curve_fit(straight_line, x, y, expected, maxfev=1000)
        stdevs = np.sqrt(np.diag(cov))

        # r-squared
        # fit values, and mean
        yhat = straight_line(x, *params)
        ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
        ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
        sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
        results['determination'] = ssreg / sstot

        return results

    def create_output_directory(self):
        dir_name = f"{self.method_name}_{self.dt}_{self.image_directory}"
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        return dir_name

    def save_classification_results(self, filename, ratios, classification):
        comment = "# image   surface-to-bulk_ratio   classification[0=normal, 1=denudated]\n"
        with open(filename, 'w') as f:
            f.write(comment)

        with open(filename, 'a') as f:
            for i, im in enumerate(self.images):
                f.write(f" {im} {ratios[i]} {classification[i]}\n")
        
            

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

    plot=True
    image_directory = "images"
    data = "MultisectionFit_2021-11-12--16-57-46.dat"
    data = "BisectionFit_2021-11-12--11-40-00.dat"

    data = "MultisectionFit_2021-11-18--15-54-07.dat"


    data = "MultisectionFit_2021-11-19--17-26-24.dat"
    data = "MultisectionFit_2021-11-23--12-45-16.dat"
    data = "MultisectionFit_2021-11-23--15-06-53.dat"
    data = "MultisectionFit_2021-11-24--17-03-29.dat"

    data = "MultisectionFit_2021-11-26--12-14-32.dat"

    data = "MultisectionFit_2021-11-26--13-01-45.dat"
    data = "MultisectionFit_2021-11-26--14-48-59.dat"

    data = "MultisectionFit_2021-11-28--17-31-50.dat"
    data = "MultisectionFit_2021-11-29--10-43-55.dat"
    data = "MultisectionFit_2021-11-29--11-57-06.dat"
    data = "MultisectionFit_2021-11-29--15-03-05.dat"
    data = "MultisectionFit_2021-11-30--14-40-08.dat"
    data = "MultisectionFit_2021-11-30--16-18-46.dat"
    classification_data = 'classification_data.dat'
    
    print(f"Analysing images from {image_directory}, using data {data}\n >with classification data {classification_data}..")


    trainers = [TrainLogisticRegression, TrainSVM]
    cc = ClassificationContainer(trainers, data, image_directory, classification_data, classify_from_file=1)
    cc.train_models(plot=plot)

    
    # for trainer in trainers:
    #     cc = ClassificationContainer()
    # # Now analyse
    # print(f"> Alpha-beta fraction")    
    # analysis = AlphaBetaFraction
    # ac = AnalysisContainer(analysis, image_directory)
    # ac.analyse_images(plot=plot)

    # print(f"> Chris Alpha-beta fraction")    
    # analysis = ChrisAlphaBetaFraction
    # ac = AnalysisContainer(analysis, image_directory)
    # ac.analyse_images(plot=plot)

    
    
    
        



