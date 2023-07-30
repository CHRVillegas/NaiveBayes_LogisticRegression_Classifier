import numpy as np
import math
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        #Setting all required variables to complete Naive Bayes 
        self.numHateWords = 0
        self.numNOTHateWords = 0
        self.hateClassCount = 0
        self.NOThateClassCount = 0
        self.probHatePrior = 0
        self.probNOTHatePrior = 0
        self.wordHateDict = {}
        self.NOTwordHateDict = {}
        

    '''
    priorProbability(X, Y):
    Detects prior probailities and aquries total word count per class and returns detected prior probabilities.
    '''
    def priorProbability(self, X, Y):

        #Set Prior Probabilities
        hateClassCount = 0
        NOThateClassCount = 0
        for i in range(len(X)):
            if Y[i] == 1:
                hateClassCount+=1
            elif Y[i] == 0:
                NOThateClassCount +=1
        
        #Get total number of words per class
        self.hateClassCount = hateClassCount
        self.NOThateClassCount = NOThateClassCount
        self.totalCount = self.hateClassCount + self.NOThateClassCount
        
        probHatePrior = self.hateClassCount / self.totalCount
        probNOTHatePrior = self.NOThateClassCount / self.totalCount

        return probNOTHatePrior, probHatePrior
    
    '''
    wordProbs(X, Y):
    Calculate Individual word Probabilities 
    '''
    def wordProbs(self, X, Y):

        #Iterate through each row and column to aquire potential probabilities
        for i in range(len(X[0])):
            for j in range(len(X)):
                if Y[j] == 1:
                    self.numHateWords += X[j][i]
                    self.wordHateDict[i] = X[j][i] + self.wordHateDict.get(i, 0)
                elif Y[j] == 0:
                    self.numNOTHateWords += X[j][i]
                    self.NOTwordHateDict[i] = X[j][i] + self.NOTwordHateDict.get(i, 0)

        
        #Implement Add-1 Smoothing for each Key found in the dictionaries for Hate and NOTHate words
        #Only need to look at keys from one of the dictionaries since they share keys
        for i, j in self.NOTwordHateDict.items():
            self.NOTwordHateDict[i] = ((self.NOTwordHateDict[i]+1) / (self.numNOTHateWords+len(X[0])))
            self.wordHateDict[i] = ((self.wordHateDict[i]+1) / (self.numHateWords+len(X[0])))

  
    '''
    fit(X, Y):
    Train Model Based on Training Sets Provided
    '''  
    def fit(self, X, Y):
        self.NOThatePrior, self.hatePrior = self.priorProbability(X, Y)
        self.wordProbs(X, Y)    
    

    '''
    predict(X):
    After model training get Log Probabilities for each word and predict overall sentence classification
    '''
    def predict(self, X):
        yPrediction = np.array([])
        for i in range(len(X)):
            logHateProb, logNotHateProb = 0, 0
            #Sum Log of word Probabilities
            for j in range(len((X[i]))):
                if X[i][j] != 0:
                    logNotHateProb += np.log(self.NOTwordHateDict[j])
                    logHateProb += np.log(self.wordHateDict[j])
            
            #Convert from Log Probabilities
            notHatePred = self.NOThatePrior * np.exp(logNotHateProb)
            hatePred = self.hatePrior * np.exp(logHateProb)

            #Apply class to sentence after probability classification
            if notHatePred >= hatePred:
                yPrediction = np.append(yPrediction, 0)
            else:
                yPrediction = np.append(yPrediction, 1)
        return yPrediction

    '''
    getwordDicts():
    Returns Dictionaries of Words containing their Index found in the Feature Extractor Unigram as well as their Potential "Hate" Scores
    '''
    def getwordDicts(self):
        return self.NOTwordHateDict, self.wordHateDict

class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        self.bias = 0.1

    '''
    calcSigmoid(row, cefficients):
    Take yHat and map to probability.
    '''
    def calcSigmoid(self, row, coefficients):
        yHat = coefficients[0]
        for i in range(len(row)-1):
            yHat += coefficients[i+1] * row[i]
        return 1.0 / (1.0+math.exp(-yHat))
    
    '''
    stochGD(X, Y, alpha, iterations, mylambda):
    Sweep through training set, and perform upadates for each training sample.
    '''
    def stochGD(self, X, Y, alpha, iterations, mylambda):
        coefficients = np.zeros(X.shape[1])
        index = 0
        for iter in range(iterations):
            index = 0
            print('>Iterations=%d' %(iter))
            for i in (X):
                prob = self.calcSigmoid(i, coefficients)
                error = i[-1] - prob
                coefficients[0] = coefficients[0] - alpha * error * (1/mylambda) * prob * (Y[index]-prob)
                for j in range(len(i)-1):
                    coefficients[j+1] = coefficients[j+1] - alpha * error * (1/mylambda) * prob * (Y[index]-prob) * i[j]
                index+=1
        
        return coefficients
    
    '''
    fit(X, Y):
    Train Model Based on Training Sets Provided
    '''  
    def fit(self, X, Y):
        iterations = 15
        alpha = 0.3
        mylambda = 1
        self.weights = self.stochGD(X, Y, alpha, iterations, mylambda)
    
    '''
    predict(X):
    After model training get predictions for each sentence in desired set
    '''
    def predict(self, X):
        predictions = np.array([])
        for row in X:
            prob = self.calcSigmoid(row, self.weights)
            predictions = np.append(predictions, round(prob))
        return predictions


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayesClassifier):
class BonusClassifier(NaiveBayesClassifier):
    def __init__(self):
        super().__init__()
