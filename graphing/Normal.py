import scipy, scipy.stats
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

np.set_printoptions(suppress=True)


def getMean():
    mu = int(input("Enter the mean: "))
    return mu

def getSigma():
    return int(input("Enter sigma: "))

mu = getMean()
sigma = getSigma()
range = np.arange(mu-5*sigma, mu+5*sigma, 0.001)
probabilities = scipy.stats.norm.pdf(range, mu, sigma)
plt.plot(range, probabilities, color="magenta", lw=2)

#format axes
#plt.axis([-(max(xTries)-min(xTries))*0.05, max(xTries)*1.05, -0.01, max(probabilities)*1.10])

xMin, xMax = mu-5*sigma, mu+5*sigma
yMin, yMax = 0, max(probabilities)*2
plt.axis([xMin, xMax, yMin, yMax])
plt.axvline(x=mu, color="black", lw=1, ls="--")
plt.axhline(y=0, color="black", lw=1, ls="--")
plt.margins(pad=4)

plt.title("Normal distribution for mu = {0} and sigma = {1}".format(mu, sigma))
plt.xlabel("Values")
plt.ylabel("Probabilities")

plt.show()


'''
SOURCES:

https://plot.ly/python/setting-graph-size/
http://www.diveintopython.net/native_data_types/lists.html#d0e5887
https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/
'''
