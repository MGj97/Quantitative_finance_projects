


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# Impostazioni per grafici di grandi dimensioni
plt.rcParams["figure.figsize"] = (14, 6)

# Generazione del rumore bianco stretto
mean = 0
std = 0.015
obs = 1000
samples = np.random.normal(mean, std, size=obs)
swn = pd.DataFrame(samples, columns=['delta'])

# Calcolo dei livelli cumulati (Random Walk)
swn['level'] = swn['delta'].cumsum()

# Grafico dei livelli esponenziali
plt.figure()
plt.plot(np.exp(swn['level']))
plt.title("Strict White Noise", fontsize=16, loc="left")
plt.tight_layout()
plt.show()

# Grafico dell'autocorrelazione sui livelli
plt.figure()
plot_acf(swn['level'], lags=23)
plt.title("Autocorrelation of Levels")
plt.tight_layout()
plt.show()

# Test ADF sui livelli
result = adfuller(swn['level'])
print('ADF Statistic test random walk: %.2f' % result[0])
print('p-value: %.3f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.2f' % (key, value))

# Grafico dell'autocorrelazione sugli incrementi (delta)
plt.figure()
plot_acf(swn['delta'], lags=23)
plt.title("Autocorrelation of Increments (Delta)")
plt.tight_layout()
plt.show()

# Test di Ljung-Box sugli incrementi
LB_lags = 23
result = acorr_ljungbox(swn['delta'], lags=LB_lags)
print("Ljung-Box test results on increments:")
print(result)

# Test di Ljung-Box sui valori assoluti degli incrementi
result_abs = acorr_ljungbox(swn['delta'].abs(), lags=LB_lags)
print("Ljung-Box test results on absolute increments:")
print(result_abs)

# Calcolo delle statistiche descrittive
M = swn['delta'].mean()
ST = swn['delta'].std()
SK = swn['delta'].skew()
KU = swn['delta'].kurtosis()

# Standardizzazione dei dati per il test KS
standardized_data = (swn['delta'] - M) / ST
KS = stats.kstest(standardized_data, 'norm')

# Istogramma degli incrementi con curva normale sovrapposta
plt.figure()
count, bins, ignored = plt.hist(swn['delta'], bins=50, density=True, color='b', alpha=0.6, label="Empirical Distribution")
plt.plot(bins, stats.norm.pdf(bins, M, ST), 'r-', linewidth=2, label="Normal Distribution")

plt.title("Probability Density Function SWN", fontsize=16, loc="left")
plt.xlabel("Delta", size=10)
plt.ylabel("Density", size=10)
plt.legend()

# Annotazioni delle statistiche
props = dict(boxstyle='round', facecolor='white', alpha=0.5)
textstr = '\n'.join((
    r'$\mathrm{Daily\ Expected\ Return}=%.3f$%%' % (M*100, ),
    r'$\mathrm{Daily\ Standard\ Deviation}=%.2f$%%' % (ST*100, ),
    r'$\mathrm{Skewness}=%.2f$' % (SK, ),
    r'$\mathrm{Excess\ Kurtosis}=%.2f$' % (KU, ),
    r'$\mathrm{KS\ Statistic}=%.3f$' % (KS.statistic, ),
    r'$\mathrm{KS\ p-value}=%.3f$' % (KS.pvalue, )))
plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Q-Q Plot utilizzando scipy.stats
plt.figure()
stats.probplot(swn['delta'], dist="norm", plot=plt)
plt.title("Q-Q Plot: Sample Quantiles vs. Theoretical Quantiles")
plt.tight_layout()
plt.show()

# Calcolo della media mobile della deviazione standard (finestra di 23)
swn['MA_23d'] = swn['delta'].rolling(window=23).std()
swn['std_dev_series'] = ST  # Serie costante con la deviazione standard globale

# Grafico della serie degli incrementi con la media mobile e le bande di deviazione standard
plt.figure()
plt.plot(swn['delta'], color='black', label='Delta')

plt.plot(swn['MA_23d'], color='blue', label='23-day Rolling STD')
plt.plot(-swn['MA_23d'], color='blue')
plt.axhline(y=ST, color='green', linestyle='dashed', label='$\sigma$')
plt.axhline(y=-ST, color='green', linestyle='dashed')

plt.title("Strict White Noise (SWN) with Rolling STD", fontsize=16, loc="left")
plt.ylabel("Delta", size=10)
plt.xlabel("Observation", size=10)
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()

# Grafico degli incrementi con bande multiple di deviazione standard
plt.figure()
plt.plot(swn['delta'], color='black', label='Delta')

plt.axhline(y=3*ST, color='red', linestyle='dashed', label='3$\sigma$')
plt.axhline(y=-3*ST, color='red', linestyle='dashed')
plt.axhline(y=3.9*ST, color='orange', linestyle='dotted', label='3.9$\sigma$')
plt.axhline(y=-3.9*ST, color='orange', linestyle='dotted')

plt.title("Strict White Noise (SWN) with Multiple STD Bands", fontsize=16, loc="left")
plt.ylabel("Delta", fontsize=10)
plt.xlabel("Observation", fontsize=10)
plt.legend(loc="upper right")
plt.grid(True)
plt.tight_layout()
plt.show()


# Numero di eccezioni comparate con i residui della distribuzione normale
normal_ex30= obs * (1-stats.norm.cdf(3))
print('Under normality assumption:')
print('Expected negative returns beyond 3 sigma ' + '{:.1f}'.format(normal_ex30))
print('Realized negative returns beyond 3 sigma ' + '{:.0f}'.format(sum(standardized_data<-3*ST)))
normal_ex39= obs * (1-stats.norm.cdf(3.9))
print('Expected negative returns beyond 3.9 sigma ' + '{:.1f}'.format(normal_ex39))
print('Realized negative returns beyond 3.9 sigma ' + '{:.0f}'.format(sum(standardized_data<-3.9*ST)))
# Numero delle obs dentro a una std. dev.
normal_in = obs * (1 - 2 * (1-stats.norm.cdf(1)))
print('Expected returns in the range +/- sigma: ''{:.0f}'.format(normal_in))
print('Realized returns in the range +/- sigma: ''{:.0f}'.format(sum(standardized_data,-ST)-sum(standardized_data>ST)))