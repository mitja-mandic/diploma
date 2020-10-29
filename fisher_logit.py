import numpy as np 
import pandas as pd 

def izracunaj_koeficiente(max_iteracij, matrika, vektor_rezultatov,zacetni_beta, epsilon=0.001):
    """
    n neodvisnih, binomskih slučajnih spremenljivk ~ Ber(pi)
    logistični model predpostavlja: logit(pi) = 1*beta_0 + x_i1*beta_1 + ... + x_ir*beta_r
    X je matrika teh koeficientov, velikosti n x r+1 (prvi stolpec same enke)
    vektor rezultatov so realizacije slučajnega vektorja
    """
    def