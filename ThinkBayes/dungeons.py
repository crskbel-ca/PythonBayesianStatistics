"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import random

import thinkbayes
import thinkplot

FORMATS = ['pdf', 'eps', 'png']


class Die(thinkbayes.PMF):
    """Represents the PMF of outcomes for a die."""

    def __init__(self, sides, name=''):
        """Initializes the die.

        sides: int number of sides
        name: string
        """

        thinkbayes.PMF.__init__(self, name=name)

        self.sides = sides
        self.name = name

        for x in range(1, sides+1):
            self.set(x, 1)
        self.normalize()


    def __key(self):
        return (self.sides)

    def __eq__(x, y):
        return x.__key() == y.__key()

    def __hash__(self):
        return hash(self.__key())


def pmfMax(pmf1, pmf2):
    """Computes the distribution of the max of values drawn from two Pmfs.

    pmf1, pmf2: Pmf objects

    returns: new Pmf
    """
    res = thinkbayes.PMF()
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            res.incr(max(v1, v2), p1 * p2)
    return res
    










def main():
    pmfDice = thinkbayes.PMF()
    pmfDice.set(Die(4), 5)
    pmfDice.set(Die(6), 4)
    pmfDice.set(Die(8), 3)
    pmfDice.set(Die(12), 2)
    pmfDice.set(Die(20), 1)
    pmfDice.normalize()

    #@fix: was unhashable error here:
    # http://stackoverflow.com/questions/10994229/how-to-make-an-object-properly-hashable
    # http://stackoverflow.com/questions/2909106/python-whats-a-correct-and-good-way-to-implement-hash

    mix = thinkbayes.PMF()
    for die, weight in pmfDice.items():
        for outcome, prob in die.items():
            mix.incr(outcome, weight * prob)

    mix = thinkbayes.makeMixture(pmfDice)

    colors = thinkplot.Brewer.getColors()
    thinkplot.hist(mix, width=0.9, color=colors[4])
    thinkplot.save(root='dungeons3',
                xlabel='Outcome',
                ylabel='Probability',
                formats=FORMATS)

    random.seed(17)

    d6 = Die(6, 'd6')

    # finding distribution of rolled-dice sum by SIMULATION
    dice = [d6] * 3
    three = thinkbayes.sampleSum(dice, 1000)
    three.name = 'sample'
    print("\n\nSAMPLING: ")
    three.print()

    # finding distribution of rolled-dice sum by ENUMERATION
    threeExact = d6 + d6 + d6
    threeExact.name = 'exact'
    print("\n\nENUMERATION:")
    threeExact.print()

    thinkplot.prePlot(num=2)
    thinkplot.pmf(three)
    thinkplot.pmf(threeExact, linestyle='dashed')
    thinkplot.save(root='dungeons1',
                xlabel='Sum of three d6',
                ylabel='Probability',
                axis=[2, 19, 0, 0.15],
                formats=FORMATS)

    thinkplot.clf()
    thinkplot.prePlot(num=1)
    
    # Note: pmf of max (best) attribute:
    bestAttribute2 = pmfMax(threeExact, threeExact)
    bestAttribute4 = pmfMax(bestAttribute2, bestAttribute2)
    bestAttribute6 = pmfMax(bestAttribute4, bestAttribute2)
    thinkplot.pmf(bestAttribute6)

    # Note: finding pmf max using efficient Cdf method:
    bestAttributeCdf = threeExact.max(6)   #@ Max() in class Cdf
    bestAttributeCdf.name = ''
    bestAttributePmf = thinkbayes.makePmfFromCdf(bestAttributeCdf)
    bestAttributePmf.print()

    thinkplot.pmf(bestAttributePmf)
    thinkplot.save(root='dungeons2',
                xlabel='Sum of three d6',
                ylabel='Probability',
                axis=[2, 19, 0, 0.23],
                formats=FORMATS)
    


if __name__ == '__main__':
    main()
