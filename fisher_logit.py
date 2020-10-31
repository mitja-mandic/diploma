import numpy as np 
import pandas as pd 

def p_i(vektor):
    return np.exp(vektor)/(1+np.exp(vektor))

def varianca(vektor):
    return np.diag(np.array(vektor) * (1 - np.array(vektor)))



def izracunaj_koeficiente(max_iteracij, matrika, vektor_rezultatov, epsilon=0.001):
    """
    n neodvisnih, bernulijevih slučajnih spremenljivk ~ Ber(pi)
    
    logistični model predpostavlja: logit(pi) = 1*beta_0 + x_i1*beta_1 + ... + x_ir*beta_r
    
    X je matrika teh koeficientov dimenzije: (n, r+1)
    vektor rezultatov so realizacije slučajnega vektorja, dimenzije: (n,)
    
    score dimenzije: (r+1,1)
    info dimenzije: (r+1,r+1)
    var dimenzije: (n,n)
    """

    #začetni podatki
    matrika = np.array(matrika)
    #print(np.shape(matrika))
    
    vektor_rezultatov = np.reshape(vektor_rezultatov, (np.shape(vektor_rezultatov)[0],))

    zacetni_beta = np.array(np.zeros(np.shape(matrika)[1]))
    #print(np.shape(zacetni_beta))

    zacetni_p = np.array(p_i(np.matmul(matrika, zacetni_beta)))
    #print(zacetni_p)
    zacetna_varianca = varianca(zacetni_p)
    #print(np.shape(zacetna_varianca))
 

    zacetni_score = np.matmul(np.transpose(matrika), (vektor_rezultatov - zacetni_p))
    #print(np.shape(vektor_rezultatov - zacetni_p))

    zacetni_info = np.matmul(np.matmul(np.transpose(matrika), zacetna_varianca),matrika)
    
    #print(np.shape(np.matmul(np.transpose(matrika), zacetna_varianca)))

    beta_star = zacetni_beta
    
    p = zacetni_p
    #print(beta_star)
    iteracije = 0
    
    #zacetni_h = np.linalg.solve(zacetni_info,zacetni_score)
    #print(zacetni_h)

    beta_nov = beta_star + np.matmul(np.linalg.inv(zacetni_info), zacetni_score)
    #print(beta_nov)
    while True:
        if iteracije - 1 > max_iteracij:
            return print('presegli ste stevilo iteracij')
        if all(np.abs(np.array(beta_star) - np.array(beta_nov)) < epsilon):
            break
        else:
            p = p_i(np.matmul(matrika, beta_nov))

            var = varianca(p)
            
            info = np.matmul(np.matmul(np.transpose(matrika), var), matrika) #matrika je (23,2) var je  (2,2)
            
            score = np.matmul(np.transpose(matrika), (vektor_rezultatov - p))

            beta_star = beta_nov
            beta_nov = beta_star + np.matmul(np.linalg.inv(info), score)
            print(beta_nov)
            iteracije += 1
    parametri = {'var-kovar' : var, 'parametri' : beta_nov}
    #print(parametri)
    return parametri


podatki = pd.read_csv("podatki.txt")
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values

rešitev = izracunaj_koeficiente(20, matrika,vrednosti_y)
