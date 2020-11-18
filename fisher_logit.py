import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

def p_i(vektor):
    return np.exp(vektor)/(1+np.exp(vektor))

def varianca(vektor, skupine):
    return np.diag(np.array(skupine) * (np.array(vektor) * (1 - np.array(vektor))))



def izracunaj_koeficiente(max_iteracij, matrika, vektor_rezultatov,vektor_skupin = [], zacetni_beta = [], epsilon=0.001):
    
    """
    
    logistični model predpostavlja: logit(pi) = 1*beta_0 + x_i1*beta_1 + ... + x_ir*beta_r
    
    X je matrika teh koeficientov dimenzije: (n, r+1)
    vektor rezultatov so realizacije slučajnega vektorja, dimenzije: (n,)
    
    score dimenzije: (r+1,1)
    info dimenzije: (r+1,r+1)
    var dimenzije: (n,n)
    """
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
    #print(np.shape(matrika))
    
    vektor_rezultatov = np.reshape(vektor_rezultatov, (n,))

    zacetni_beta = np.array(np.zeros(np.shape(matrika)[1]))
    #print(np.shape(zacetni_beta))

    zacetni_p = np.array(p_i(np.matmul(matrika, zacetni_beta)))
    #print(zacetni_p)
    zacetna_varianca = varianca(zacetni_p, vektor_skupin)
    
    print(np.shape(zacetna_varianca))
 

    zacetni_score = np.matmul(np.transpose(matrika), (vektor_rezultatov -  vektor_skupin * zacetni_p))
    #print(np.shape(vektor_rezultatov - zacetni_p))

    zacetni_info = np.matmul(np.matmul(np.transpose(matrika), zacetna_varianca),matrika)
    
    #print(np.shape(np.matmul(np.transpose(matrika), zacetna_varianca)))

    beta_star = zacetni_beta
    
    p = zacetni_p
    #print(beta_star)
    iteracije = 0
    
    zacetni_h = np.linalg.solve(zacetni_info,zacetni_score)
    #print(zacetni_h)

    #beta_nov = beta_star + np.matmul(np.linalg.inv(zacetni_info), zacetni_score)
    beta_nov = beta_star + zacetni_h
    #print(beta_nov)
    while True:
        if iteracije - 1 > max_iteracij:
            return print('presegli ste stevilo iteracij')
        if all(np.abs(np.array(beta_star) - np.array(beta_nov)) < epsilon):
            break
        else:
            p = p_i(np.matmul(matrika, beta_nov)) # n * (r+1) operacij
            var = varianca(p, vektor_skupin)
            print(np.shape(matrika), np.shape(var))
            #print(np.shape(var))
            info = np.matmul(np.matmul(np.transpose(matrika), var), matrika) #matrika je (23,2) var je  (23,23). produkt 2x23 * 23x23 * 23x2
            #v info množiš r+1xn * nxn * nxr+1

            #print(np.shape(info))
            score = np.matmul(np.transpose(matrika), (vektor_rezultatov - vektor_skupin * p)) # r+1xn * nx1

            h = np.linalg.solve(info, score) #r+1xr+1

            beta_star = beta_nov
            #beta_nov = beta_star + np.matmul(np.linalg.inv(info), score)
            beta_nov = beta_star + h
            #print(beta_nov)
            iteracije += 1
    parametri = {'var-kovar' : var, 'parametri' : beta_nov, 'p':p}
    #print(parametri)
    return parametri


podatki = pd.read_csv("podatki.txt")
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values
#print(podatki[['dodatna']])
#print(podatki.columns)
resitev = izracunaj_koeficiente(20, matrika,vrednosti_y)

#podatki_hrosci = pd.read_csv('SC1_11_beetles.txt')
#skupine = podatki_hrosci[['n']].values
#podatki_hrosci['dodatna'] = 1
#conc_kvadrat = podatki_hrosci[['conc']].pow(2)
#podatki_hrosci['conc_kvadrat'] = conc_kvadrat
#matrika = podatki_hrosci[['dodatna','conc','conc_kvadrat']].values
#
#vrednost_y = podatki_hrosci[['y']].values
#
#beta_ena = np.full((3,),np.log(sum(vrednost_y)/sum(skupine-vrednost_y)))
#
#resitev = izracunaj_koeficiente(20, matrika, vrednost_y, skupine)
#
fitted = resitev['p']

intercept = resitev['parametri'][0]
slope = resitev['parametri'][1]
#slope1 = resitev['parametri'][2]


#print(fitted)

def main():
    x = np.linspace(40,90,1000)
    #y = np.exp(intercept + x * slope + (x ** 2) * slope1) / (1 + np.exp(intercept + x * slope + (x ** 2) * slope1))
    y = np.exp(intercept + x * slope)/ (1 + np.exp(intercept + x * slope))
    #plt.figure()
    plt.plot(x, y)
    #plt.xlabel('$x$')
    #plt.ylabel('$\exp(x)$')

    #plt.plot(podatki_hrosci[['conc']].values,fitted, 'o')
    plt.plot(podatki[['TEMPERATURE']].values,fitted, 'o')
    plt.plot(podatki[['TEMPERATURE']].values, podatki[['O_RING_FAILURE']].values,'o')


    #plt.figure()
    #plt.plot(x, -np.exp(-x))
    #plt.xlabel('$x$')
    #plt.ylabel('$-\exp(-x)$')

    plt.show()

if __name__ == '__main__':
    main()