"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from dice import Dice
import thinkplot

class Train(Dice):
    """The likelihood function for the train problem is the same as
    for the Dice problem."""


def mean(suite):
    total = 0
    for hypo, prob in suite.items():
        total += hypo * prob
    return total


def makePosterior(high, dataset):
    hypos = range(1, high+1)
    suite = Train(hypos)
    suite.name = str(high)

    for data in dataset:
        suite.update(data)

    thinkplot.pmf(suite)
    return suite


def main():
    # ASSUME: we add more data: meaning we also see locomotives 30 and 90, alongside 60.
    dataset = [30, 60, 90]

    # high = sample size  (total number of locomotives).
    for high in [500, 1000, 2000]:
        suite = makePosterior(high, dataset)
        print(high, "; ", suite.mean(), sep="")

    thinkplot.save(root='train2',
                   xlabel='Number of trains',
                   ylabel='Probability')


if __name__ == '__main__':
    main()
