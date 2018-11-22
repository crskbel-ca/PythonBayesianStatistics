"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import sys

import matplotlib.pyplot as pyplot
import thinkbayes

from src import thinkplot

# TODO hee need help
'''
LEFT OFF HERE FOR MAKING FUNCTIONS CAMELCASE. # TODO

'''


#, 'eps', 'png'
FORMATS = ['pdf']


def strafingSpeed(alpha, beta, x):
    """Computes strafing speed, given location of shooter and impact.

    alpha: x location of shooter
    beta: y location of shooter
    x: location of impact

    Returns: derivative of x with respect to theta
    """
    # NOTE: strafing speed is speed of moving x at a given angle theta.
    theta = math.atan2(x - alpha, beta)
    speed = beta / math.cos(theta)**2
    return speed


def makeLocationPmf(alpha, beta, locations):
    """Computes the Pmf of the locations, given alpha and beta.

    Given that the shooter is at coordinates (alpha, beta),
    the probability of hitting any spot is inversely proportionate
    to the strafe speed.

    alpha: x position
    beta: y position
    locations: x locations where the pmf is evaluated

    Returns: Pmf object
    """
    pmf = thinkbayes.PMF()
    for x in locations:
        # NOTE: prob of hitting any location on the wall is inversely related to strafing speed
        prob = 1.0 / strafingSpeed(alpha, beta, x)
        pmf.set(x, prob)
    pmf.normalize()
    return pmf


class Paintball(thinkbayes.Suite, thinkbayes.Joint):
    """Represents hypotheses about the location of an opponent."""

    def __init__(self, alphas, betas, locations):
        """Makes a joint suite of parameters alpha and beta.

        Enumerates all pairs of alpha and beta.
        Stores locations for use in Likelihood.

        alphas: possible values for alpha
        betas: possible values for beta
        locations: possible locations along the wall
        """
        self.locations = locations
        pairs = [(alpha, beta)
                 for alpha in alphas
                 for beta in betas]
        thinkbayes.Suite.__init__(self, pairs)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: pair of alpha, beta
        data: location of a hit

        Returns: float likelihood
        """
        alpha, beta = hypo # note - hypothetical shooter coordinates
        x = data # note - location of observed splash (paintball hit)
        pmf = makeLocationPmf(alpha, beta, self.locations) # pmf: for all locations, at this particular (alpha,beta) pair.
        like = pmf.prob(x)
        return like


def makePmfPlot(alpha = 10):
    """Plots Pmf of location for a range of betas."""
    locations = range(0, 31)

    betas = [10, 20, 40]
    thinkplot.prePlot(num=len(betas))

    for beta in betas:
        pmf = makeLocationPmf(alpha, beta, locations)
        pmf.name = 'PMF(alpha = 10, beta = %d)' % beta
        thinkplot.pmf(pmf)

    thinkplot.save('paintball1_pmfOfLocationForRangeOfBetas',
                   xlabel='Distance',
                   ylabel='Prob',
                   formats=FORMATS)


def makePosteriorPlot(suite):
    """Plots the posterior marginal distributions for alpha and beta.

    suite: posterior joint distribution of location
    """
    marginal_alpha = suite.marginal(0)
    marginal_alpha.name = 'PMF(alpha)'
    marginal_beta = suite.marginal(1)
    marginal_beta.name = 'PMF(beta)'

    print("\n(ALPHA, BETA): Shooter locations: ")
    print('alpha CI', marginal_alpha.credibleInterval(50))
    print('beta CI', marginal_beta.credibleInterval(50))

    thinkplot.prePlot(num=2)

    #thinkplot.Pmf(marginal_alpha)
    #thinkplot.Pmf(marginal_beta)

    thinkplot.cdf(thinkbayes.makeCdfFromPmf(marginal_alpha))
    thinkplot.cdf(thinkbayes.makeCdfFromPmf(marginal_beta))

    thinkplot.save('paintball2_posteriorMarginalDist_for_alpha_and_beta',
                   xlabel='Distance',
                   ylabel='Prob',
                   loc=4,
                   formats=FORMATS)


def makeConditionalPlot(suite):
    """Plots marginal CDFs for alpha conditioned on beta.

    suite: posterior joint distribution of location
    """
    betas = [10, 20, 40]
    thinkplot.prePlot(num=len(betas))

    for beta in betas:
        cond = suite.conditional(0, 1, beta) # i=0 (variable we want), j=1 (variable cond), val=beta (value that jth variable must have)
        cond.name = 'PMF(alpha | beta = %d)' % beta
        thinkplot.pmf(cond)

    # THe posterior conditional marginal distributions
    thinkplot.save('paintball3_marginalCDFS_for_alpha|beta',
                   xlabel='Distance',
                   ylabel='Prob',
                   formats=FORMATS)


def makeContourPlot(suite):
    """Plots the posterior joint distribution as a contour plot.

    suite: posterior joint distribution of location
    """
    thinkplot.contour(suite.getDict(), contour=False, pcolor=True)

    thinkplot.save('paintball4_posteriorJointAsContour',
                   xlabel='alpha',
                   ylabel='beta',
                   axis=[0, 30, 0, 20],
                   formats=FORMATS)


def makeCrediblePlot(suite):
    """Makes a plot showing several two-dimensional credible intervals.

    suite: Suite
    """
    d = dict((pair, 0) for pair in suite.keys())

    percentages = [75, 50, 25]
    for p in percentages:
        interval = suite.maxLikeInterval(p)
        for pair in interval:
            d[pair] += 1 # note: maps each pair to number of intervals it appears in.

    thinkplot.contour(d, contour=False, pcolor=True)
    pyplot.text(17, 4, '25', color='white')
    pyplot.text(17, 15, '50', color='white')
    pyplot.text(17, 30, '75')

    thinkplot.save('paintball5_credibleintervals',
                   xlabel='alpha',
                   ylabel='beta',
                   formats=FORMATS)


def main(script):

    alphas = range(0, 31) # upper bound = max room width
    betas = range(1, 51) # upper bound = max room length
    locations = range(0, 31)

    suite = Paintball(alphas, betas, locations) # Note: choosing the uniform prior because all spots in room are equally likely
    suite.updateSet([15, 16, 18, 21])
    # for each data splash location (15,16..),
    #   for each (alpha,beta) shooter coord pair,
    #       create new likelihood pmf
    #       find likpmfprob at x (splash loc)
    #       update prior at key (alpha,beta) with this like value


    makePmfPlot()

    #note: alpha = shooter most likely at 18 feet,
    #note: beta = shooter most likely at 10 feet, no difference in shooting success above 10
    makePosteriorPlot(suite)

    # note: we can tell alpha and beta are dependent because for larger betas, alpha is wider.
    makeConditionalPlot(suite)

    makeCrediblePlot(suite)

    makeContourPlot(suite)




if __name__ == '__main__':
    main(*sys.argv)
