#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: 
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)

    FeatureVector = defaultdict(int)  # create a feature vector(Show FV as below) with value 0.

    w_lst = x.split()  # split string x and store in a list.
    for word in w_lst:  # loop over the word list
        if word not in FeatureVector:  # If word is not in FV, add into the dic and value be 1.
            FeatureVector[word] = 1
        else:
            FeatureVector[word] += 1

    return FeatureVector

    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # feature => weight

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)

    def predictor(x):
        """
        @param x: dictionary with review.
        @return: score between x and weight.
        """
        x = featureExtractor(x)  # Do the feature extract of any review.
        return 1 if dotProduct(weights, x) > 0 else -1  # Calculate the k

    for epochs in range(numEpochs):
        for t_x, t_y in trainExamples:
            if t_y < 0:  # Change the label in trainExample from -1 to 0, else remain 1.
                t_y = 0
            else:
                t_y = 1

            review_dict = featureExtractor(t_x)  # Build feature extractor
            t_k = dotProduct(weights, review_dict)  # Calculate k
            h = 1 / (1+math.exp(-t_k))  # Calculate h
            increment(d1=weights, scale=(-alpha * (h - t_y)), d2=review_dict)  # Calculate weights

        t_error = evaluatePredictor(trainExamples, predictor)  # Evaluate the error from train example.
        print('Training Error: (' + str(epochs) + ' epoch): ' + str(t_error))  # Print the train error.
        v_error = evaluatePredictor(validationExamples, predictor) # Evaluate the error from validation example.
        print('Validation Error: (' + str(epochs) + ' epoch): ' + str(v_error))  # Print the train error.

    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and value is exactly 1.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
        phi = {}  # Empty dictionary to store the random key and value from weight.
        y = 0  # Default value.
        all_weight_key = []  # Empty list to store all the key in weight.

        for key in weights:  # Add all the key in weight into a list.
            all_weight_key.append(key)

        r = random.randint(1, len(weights))  # Random amount of words.
        for i in range(r):
            phi[all_weight_key[i]] = 1  # Add the key from weights into phi.
            y = dotProduct(phi, weights)  # Do the dot between phi and weight.
            if y >= 0:  # When y greater than 0, which is a good review. y = 1, else, bad review, y = -1.
                phi[all_weight_key[i]] = 1
                y = 1
            else:
                phi[all_weight_key[i]] = -1
                y = -1

        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        ch_review = ''  # To store all the character from a review.
        ch_dictionary = {}  # Character dictionary.

        for ch in x:  # Loop over the character in a review.
            if ch != ' ':
                ch_review += ch

        for i in range(len(ch_review)-n+1):  # Loop over the ch_review to build the dictionary.
            if ch_review[i:i+n] not in ch_dictionary:
                ch_dictionary[ch_review[i:i+n]] = 1
            else:
                ch_dictionary[ch_review[i:i + n]] += 1

        return ch_dictionary

        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))

