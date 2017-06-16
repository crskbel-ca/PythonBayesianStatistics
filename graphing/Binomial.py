import scipy, scipy.stats
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

np.set_printoptions(suppress=True)


def getX():
    xTries = int(input("Enter x: "))
    return range(xTries+1)

def getN():
    return int(input("Enter n: "))


def getP():
    return float(input("Enter p: "))

n = getN()
xTries = range(n + 1)
p = getP()
probabilities = scipy.stats.binom.pmf(xTries, n, p)
print(probabilities)
plt.plot(xTries, probabilities, "o", color="magenta")

#format axes
#plt.axis([-(max(xTries)-min(xTries))*0.05, max(xTries)*1.05, -0.01, max(probabilities)*1.10])
xMin, xMax = (min(xTries) - 2), (max(xTries) + 2)
yMin, yMax = (min(probabilities) - 0.1), (max(probabilities) + 0.1)
plt.axis([xMin, xMax, yMin, yMax])
plt.margins(pad=4)

plt.title("Distributions distribution for n = {0} and p = {1}".format(n, p))
plt.xlabel("xTries")
plt.ylabel("Probabilities")

plt.draw()
plt.show()


'''
SOURCES:

https://plot.ly/python/setting-graph-size/
http://www.diveintopython.net/native_data_types/lists.html#d0e5887
https://oneau.wordpress.com/2011/02/28/simple-statistics-with-scipy/
'''
