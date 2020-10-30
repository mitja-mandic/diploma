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
    X je matrika teh koeficientov, velikosti n x r+1 (prvi stolpec same enke)
    vektor rezultatov so realizacije slučajnega vektorja
    """

    #začetni podatki
    matrika = np.array(matrika)
    #print(np.shape(matrika))
    #vektor_rezultatov = np.array(vektor_rezultatov)

    zacetni_beta = np.array(np.zeros(np.shape(matrika)[1]))
    #print(np.shape(zacetni_beta))

    zacetni_p = np.array(p_i(np.matmul(matrika, zacetni_beta)))
 

    zacetna_varianca = varianca(zacetni_p)
    #print(np.shape(zacetna_varianca))
 

    zacetni_score = np.matmul(np.transpose(matrika), (vektor_rezultatov - zacetni_p)) #problem ker en (N,1) drugi pa (N,)
    print(np.shape(vektor_rezultatov - zacetni_p))

    zacetni_info = np.matmul(np.matmul(np.transpose(matrika), zacetna_varianca),matrika)
    print(np.shape(zacetni_info))
    #varianca = zacetna_varianca
    #score = zacetni_score
    #info = zacetni_info
    beta_star = zacetni_beta
    print(beta_star)
    iteracije = 0
    #zacetni_h = np.linalg.solve(zacetni_info,zacetni_score)
    #print(zacetni_h)

    beta_nov = beta_star + np.matmul(np.linalg.inv(zacetni_info), zacetni_score)
    
    while True:

        if iteracije > max_iteracij:
            return print('presegli ste stevilo iteracij')
        elif all(np.abs(beta_star - beta_nov)) < epsilon:
            break
        else:
            var = varianca(beta_nov)
            info = np.matmul(np.matmul(np.transpose(matrika), var), matrika)
            score = np.matmul(np.transpose, (vektor_rezultatov - np.matmul(matrika, p_i(beta_nov))))

            h = np.linalg.solve(info, score)

            beta_star = beta_nov
            beta_nov = beta_star + h
            
            iteracije += 1
    parametri = {'var-kovar' == var, 'parametri' == beta_nov}
    return parametri


podatki = pd.read_csv("podatki.txt")
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values

rešitev = izracunaj_koeficiente(20, matrika,vrednosti_y)