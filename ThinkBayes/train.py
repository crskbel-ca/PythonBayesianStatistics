"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from dice import Dice
import thinkplot


class Train(Dice):
    """Represents hypotheses about how many trains the company has.

    The likelihood function for the train problem is the same as
    for the Dice problem.
    """


def main():
    # Step 1: Prior P(N) = 1/1000
    hypos = range(1, 1001) # assume there are 1 - 1000 locomotives in the train, given we saw the number 60
    suite = Train(hypos)

    suite.update(60)
    print(suite.mean())

    thinkplot.prePlot(1)
    thinkplot.pmf(suite)
    thinkplot.save(root='train1',
                   xlabel='Number of trains',
                   ylabel='Probability',
                   formats=['pdf', 'eps'])
    thinkplot.show()


if __name__ == '__main__':
    main()
