# Pricing di una Call/Put Europea/Americana - modello CCR

import math as mt
import numpy as np

# definire imput
# -- mercato/contratto
r=0.001
S0=123
K=130
T=2

flagUE=1 # 0 Europea, 1 Americana
flagCP=-1 #1 Call, -1 Put
# -- calibrazione
sigma=0.5
# -- discretizzazione (intervalli tra 0 e T)
M=1000

# definiamo le variabili
dt=T/M
u=mt.exp( sigma* mt.sqrt(dt))
d=1/u
disc=mt.exp(-r*dt) #fattore di sconto - cap esponenziale
#disc=1/(1+r)**dt #fattore di sconto - cap composta
cap=1/disc #fattore di capitalizzazione
#Risk-Neutral probability
pu=( cap - d)/(u - d)
pd= 1-pu

# Check assenza di arbitraggio
print('d < cap < u ?')
print(d)
print(cap)
print(u)

# Creo matrix e inserisco il payoff
albero=np.zeros((M+1,M+1)) # Creo una matrice (array bidimensionale) di dimensione (M+1) x (M+1) piena di zeri
for i in range (0, M+1): # Itero attraverso gli indici da 0 a M inclusi
  # Calcolo il valore dell'asset sottostante al tempo M-i (up. Mentre i = down)
  ST=S0*(u**(M-i))*(d**i)
  # Assegno alla cella nella posizione (i, -1) (payoff) il valore massimo tra flagCP*(ST-K) e 0
  albero[i,-1]=max( flagCP*(ST-K),0)
#print(albero)

# Pricing
IV=0 # Valore intrinseco
for j in range(M-1, -1, -1): # Questo loop itera attraverso gli indici da M-1 a 0 (incluso). j rappresenta il livello temporale nell'albero binomiale.
  for i in range(0, j+1): # da 0 a j (incluso). i rappresenta il nodo all'interno del livello temporale j.
    if flagUE==1: # Americana
      S=S0*(u**(j-i))*(d**(i))
      IV=flagCP*(S-K)
    CV=disc*(pu*albero[i,j+1]+pd*albero[i+1,j+1]) # albero[1,j+1] perchè u sta nella stessa riga ma si muove di una colonna, stesso ragionamento per d che sta una riga sotto [i+1, j+1]
    #CV è il valore continuativo dell'opzione, calcolato come il valore attuale scontato (disc) della combinazione lineare dei payoff attesi dai nodi successivi
    albero[i,j]=max(IV,CV)
prezzo=albero[0,0] #il prezzo sta nella posizione [0,0]
print(prezzo)

# Check B&S
from math import log, sqrt, exp
from scipy.stats import norm

def d1(S,K,r,T,sigma):
  return(log(S/K)+(r+sigma**2/2.)*T)/(sigma*sqrt(T))

def d2(S,K,r,T,sigma):
  return d1(S,K,r,T,sigma)-sigma*sqrt(T)

prezzoBS=S0*norm.cdf(d1(S0,K,r,T,sigma))-K*exp(-r*T)*norm.cdf(d2(S0,K,r,T,sigma))
if flagCP==-1: # put-call parity
  prezzoBS=prezzoBS-S0+K*exp(-r*T)
print(prezzoBS)

