from fisher_logit import izracunaj_koeficiente
from newton_probit import izracunaj_koeficiente_probit
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import statsmodels

cur_path = os.path.dirname(__file__)

new_path = os.path.relpath('..\\podatki\\podatki.txt', cur_path)


podatki = pd.read_csv("diploma\\koda\\podatki.txt")
podatki['dodatna'] = 1
matrika = podatki[['dodatna','TEMPERATURE']].values
vrednosti_y = podatki[['O_RING_FAILURE']].values
#print(podatki[['dodatna']])
#print(podatki.columns)
resitev_probit = izracunaj_koeficiente_probit(50, matrika,vrednosti_y)
resitev_logit = izracunaj_koeficiente(50, matrika, vrednosti_y)
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

#fitted = resitev['p']
intercept_probit = resitev_probit['parametri'][0]
slope_probit = resitev_probit['parametri'][1]

intercept_logit = resitev_logit['parametri'][0]
slope_logit = resitev_logit['parametri'][1]

#slope1 = resitev['parametri'][2]
#print(intercept,slope,slope1)

def f(x, y):
    return np.exp(intercept + x * slope + y * slope1)/ (1 + np.exp(intercept + x * slope + y * slope1))

def main():
    x = np.linspace(30,100,1000)
    y = np.linspace(50,200,1000)
    X,Y = np.meshgrid(x,y)
    y_probit = np.exp(intercept_probit + x * slope_probit)/ (1 + np.exp(intercept_probit + x * slope_probit))
    y_logit = np.exp(intercept_logit + x * slope_logit)/ (1 + np.exp(intercept_logit + x * slope_logit))
    #Z = f(X,Y)
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot_surface(X, Y, Z,cmap='viridis',edgecolor='none')
    
    
    
    
    
    plt.figure()
    plt.plot(x, y_logit, label = 'Logit')
    plt.plot(x,y_probit, label = "probit")
    plt.xlabel('$x$')
    #plt.ylabel('$\exp(x)$')

    #plt.plot(podatki_hrosci[['conc']].values,fitted, 'o')
    #plt.plot(podatki[['TEMPERATURE']].values,fitted, 'o',label = 'Izračunane verjetnosti')
    #plt.plot(podatki[['TEMPERATURE']].values, podatki[['O_RING_FAILURE']].values,'o', label = 'Podatki')
    #plt.legend()

    #plt.figure()
    #plt.plot(x, -np.exp(-x))
    #plt.xlabel('$x$')
    #plt.ylabel('$-\exp(-x)$')
    plt.show()

if __name__ == '__main__':
    main()



#
#x = np.linspace(-6, 6, 30)
#y = np.linspace(-6, 6, 30)
#
#X, Y = np.meshgrid(x, y)
#Z = f(X, Y)
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.contour3D(X, Y, Z, 50, cmap='binary')
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z')
#plt.show()

#statsmodels.discrete.discrete_model.Probit(vrednosti_y, matrika)
#result_3 = statsmodels.discrete_model.Probit
#print(result_3.summary())