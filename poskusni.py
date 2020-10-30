import numpy as np 
import pandas as pd 
#import fisher_logit

a = np.array([1, 2, 3])
def p_i(vektor):
    return np.exp(vektor)/(1+np.exp(vektor))
def varianca(vektor):
    return np.diag(np.array(vektor) * (1 - np.array(vektor)))


b = np.array([[1],[4],[1]])
c = np.array([[1,1,1],[2,3,5],[2,3,4]])
#np.reshape(b,(3,))