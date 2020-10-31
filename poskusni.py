import numpy as np 
import pandas as pd 
#import fisher_logit

a = np.array([1, 2, 3])
def p_i(vektor):
    return np.exp(vektor)/(1+np.exp(vektor))
def varianca(vektor):
    return np.diag(np.array(vektor) * (1 - np.array(vektor)))

w = p_i(a) * p_i(1-a)
b = np.array([[1],[2],[3]])
y = p_i(b) * p_i(1-b)
print(np.diag(y[:,0]))
#print(varianca(a))