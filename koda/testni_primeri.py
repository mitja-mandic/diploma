from fisher_logit import izracunaj_koeficiente
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os

cur_path = os.path.dirname(__file__)

new_path = os.path.relpath('..\\podatki\\podatki.txt', cur_path)


podatki = pd.read_csv('koda\\podatki.txt')
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values
#print(podatki[['dodatna']])
#print(podatki.columns)
resitev = izracunaj_koeficiente(20, matrika,vrednosti_y)

#podatki_hrosci = pd.read_csv('podakti/SC1_11_beetles.txt')
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

fitted = resitev['p']
intercept = resitev['parametri'][0]
slope = resitev['parametri'][1]
print(intercept,slope)
#slope1 = resitev['parametri'][2]




#print(fitted)

def main():
    x = np.linspace(40,90,1000)
    #y = np.exp(intercept + x * slope + (x ** 2) * slope1) / (1 + np.exp(intercept + x * slope + (x ** 2) * slope1))
    y = np.exp(intercept + x * slope)/ (1 + np.exp(intercept + x * slope))
    #plt.figure()
    plt.plot(x, y, label = 'Logit')
    #plt.xlabel('$x$')
    #plt.ylabel('$\exp(x)$')

    #plt.plot(podatki_hrosci[['conc']].values,fitted, 'o')
    plt.plot(podatki[['TEMPERATURE']].values,fitted, 'o',label = 'Izračunane verjetnosti')
    plt.plot(podatki[['TEMPERATURE']].values, podatki[['O_RING_FAILURE']].values,'o', label = 'Podatki')
    plt.legend()

    #plt.figure()
    #plt.plot(x, -np.exp(-x))
    #plt.xlabel('$x$')
    #plt.ylabel('$-\exp(-x)$')

    #plt.show()

if __name__ == '__main__':
    main()