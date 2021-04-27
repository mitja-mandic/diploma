import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm



def izracunaj_koeficiente_probit(max_iteracij, matrika, vektor_rezultatov,vektor_skupin = [], zacetni_beta = [], epsilon=0.001):
        
    """
    probit model predpostavlja: p_i = phi(1*beta_0 + x_i1*beta_1 + ... + x_ir*beta_r)
    
    X je matrika teh koeficientov dimenzije: (n, r+1)
    vektor rezultatov so realizacije slučajnega vektorja, dimenzije: (n,)
    
    score dimenzije: (r+1,1)
    info dimenzije: (r+1,r+1)
    var dimenzije: (n,n)
    """
    def p_i(x):
        return np.array(norm.cdf(x))
    
    def score_koef(y,m,x):
        return np.divide((y-p_i(x) * m), p_i(x)*(1-p_i(x))) * norm.pdf(x)
    
    def hessian_koef(y,m,x):
        phi = norm.cdf(x)
        koef = np.divide(norm.pdf(x),(phi*(1-phi))) * ( norm.cdf(x) * np.divide((2*phi*y - m * (phi)**2 - y),phi*(1-phi))-x * (y-m * phi))
        return np.diag(koef)


    n = np.shape(vektor_rezultatov)[0]
    r_plus1 = np.shape(matrika)[1]

    if not any(np.array(vektor_skupin)):
        vektor_skupin = np.ones((np.shape(vektor_rezultatov)[0],))
    else:
        vektor_skupin = np.reshape(vektor_skupin, (n,))
    
    if not any(np.array(zacetni_beta)):
        zacetni_beta = np.array(np.zeros(np.shape(matrika)[1]))
    else:
        zacetni_beta = np.reshape(zacetni_beta, (r_plus1,))
    
    #začetni podatki
    matrika = np.array(matrika)
    
    vektor_rezultatov = np.reshape(vektor_rezultatov, (n,))
    
    zacetni_x = np.array(np.matmul(matrika, zacetni_beta))
    
    zacetni_p = np.array(p_i(zacetni_x))
    
    zacetni_score = np.matmul(np.transpose(matrika),score_koef(vektor_rezultatov,vektor_skupin,zacetni_x))

    zacetni_hessian = np.matmul(np.matmul(np.transpose(matrika),hessian_koef(vektor_rezultatov,vektor_skupin,zacetni_x)),matrika)
    
    beta_star = zacetni_beta
    
    p = zacetni_p
    #print(beta_star)
    iteracije = 0
    
    zacetni_h = np.linalg.solve(zacetni_hessian,zacetni_score)
    
    beta_nov = beta_star + zacetni_h
    print(beta_star)
    print(beta_nov)
    while True:
        if iteracije - 1 > max_iteracij:
            return print('presegli ste stevilo iteracij')
        #elif all(np.abs(np.array(beta_star) - np.array(beta_nov)) < epsilon):
        elif np.all(np.abs(np.array(beta_star) - np.array(beta_nov)) < epsilon):
            break
        else:
            x = np.matmul(matrika, beta_nov)
            
            p = p_i(x)

            score = np.matmul(np.transpose(matrika),score_koef(vektor_rezultatov,vektor_skupin,x))
            hessian = np.matmul(np.matmul(np.transpose(matrika),hessian_koef(vektor_rezultatov,vektor_skupin,x)),matrika)
            
            #print(np.shape(info))
            h = np.linalg.solve(hessian, score) #r+1xr+1

            beta_star = beta_nov
            #beta_nov = beta_star + np.matmul(np.linalg.inv(info), score)
            beta_nov = beta_star - h
            #print(beta_nov)
            iteracije += 1
    parametri = {'parametri': beta_nov, 'p':p}
    print(iteracije)
    return parametri

podatki = pd.read_csv("diploma\\koda\\podatki.txt")
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE','PRESSURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values
#print(podatki[['dodatna']])
#print(podatki.columns)
resitev = izracunaj_koeficiente_probit(1000, matrika,vrednosti_y)
print(resitev, "probit")