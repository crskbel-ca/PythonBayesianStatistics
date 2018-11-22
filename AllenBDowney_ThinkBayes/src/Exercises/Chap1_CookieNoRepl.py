"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from src.thinkbayes import Hist
from src.thinkbayes import PMF


class Bowl:
    def __init__(self, num, vanilla, choc):
        self.number = num
        self.probVanilla = vanilla
        self.probChocolate = choc

    def getName(self): return self.number
    def probabilityOfVanilla(self): return self.probVanilla
    def probabilityOfChocolate(self): return self.probChocolate



"""
class CookieFlavor:

    def __init__(self, flavor):
        self.flavor = flavor

    def getFlavor(self): return self.flavor

"""


## todo help here is solution but can't figure out why error occurs:
# todo: https://github.com/AllenDowney/ThinkBayes2/blob/master/code/chap02soln.ipynb

class Cookie(PMF): # Cookie extends Pmf
    """A map from string bowl ID to probability."""

    def __init__(self, hypos):
        """Initialize self.

        hypos: sequence of string bowl IDs
        """
        PMF.__init__(self)

        for hypo in hypos:
            self.set(hypo, 1)
        self.normalize()



    def update(self, pmf, data):
        """Updates the PMF with new data.

        data: string cookie type
        """
        for hypo in pmf:
            pmf[hypo] *= self.likelihood(hypo, data)
        return pmf.normalize()


    def likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.

        data: string cookie type
        hypo: string bowl ID


        mix = self.hypos[hypo]
        like = mix[data]
        return like
        """
        like = hypo[data] / hypo.total()
        if like:
            hypo[data] -= 1
        return like




def main():
    """
    bowl1 = Bowl(1, vanilla=0.75, choc=0.25)
    bowl2 = Bowl(2, vanilla=0.5, choc=0.5)
    #hypos = [bowl1, bowl2]

    pmf = Cookie(bowl1, bowl2)
    pmf.print()

    pmf.update("vanilla")
    pmf.print()
    pmf.update("chocolate")
    pmf.print()
    """
    bowl1 = Hist(dict(vanilla=30, chocolate=10))
    bowl2 = Hist(dict(vanilla=20, chocolate=20))
    bowl1.printSuite()
    bowl2.printSuite()

    pmf = PMF([bowl1, bowl2])
    cookie = Cookie(pmf)
    pmf.printSuite()
    cookie.printSuite()

    # if we get 10 more chocolate cookies then we run out (without replacement)
    for i in range(10):
        cookie.update("chocolate")
        #print(cookie[bowl1])



if __name__ == '__main__':
    main()
