"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math
import numpy
import pickle
import numpy
import random
import scipy

import brfss

import thinkplot
import thinkbayes
import thinkstats

import matplotlib.pyplot as pyplot


NUM_SIGMAS = 1

class Height(thinkbayes.Suite, thinkbayes.Joint):
    """Hypotheses about parameters of the distribution of height."""

    def __init__(self, mus, sigmas, name=''):
        """Makes a prior distribution for mu and sigma based on a sample.

        mus: sequence of possible mus
        sigmas: sequence of possible sigmas
        name: string name for the Suite
        """
        pairs = [(mu, sigma) 
                 for mu in mus
                 for sigma in sigmas]

        thinkbayes.Suite.__init__(self, pairs, name=name)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Args:
            hypo: tuple of hypothetical mu and sigma
            data: float sample

        Returns:
            likelihood of the sample given mu and sigma
        """
        x = data
        mu, sigma = hypo
        like = scipy.stats.norm.pdf(x, mu, sigma)
        return like

    def logLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        Args:
            data: a list of values
            hypo: tuple of hypothetical mu and sigma

        Returns:
            log likelihood of the sample given mu and sigma (unnormalized)
        """
        x = data
        mu, sigma = hypo
        loglike = evalGaussianLogPdf(x, mu, sigma)
        return loglike

    def logUpdateSetFast(self, data):
        """Updates the suite using a faster implementation.

        Computes the sum of the log likelihoods directly.

        Args:
            data: sequence of values
        """
        xs = tuple(data)
        n = len(xs)

        for hypo in self.keys():
            mu, sigma = hypo
            total = summation(xs, mu)
            loglike = -n * math.log(sigma) - total / 2 / sigma**2
            self.incr(hypo, loglike)

    def logUpdateSetMeanVar(self, data):
        """Updates the suite using ABC and mean/var.

        Args:
            data: sequence of values
        """
        xs = data
        n = len(xs)

        m = numpy.mean(xs)
        s = numpy.std(xs)

        self.logUpdateSetABC(n, m, s)

    def logUpdateSetMedianIPR(self, data):
        """Updates the suite using ABC and median/iqr.

        Args:
            data: sequence of values
        """
        xs = data
        n = len(xs)

        # compute summary stats
        median, s = medianS(xs, num_sigmas=NUM_SIGMAS)
        print('median, s', median, s)

        self.logUpdateSetABC(n, median, s)

    def logUpdateSetABC(self, n, m, s):
        """Updates the suite using ABC.

        n: sample size
        m: estimated central tendency
        s: estimated spread
        """
        for hypo in sorted(self.keys()):
            mu, sigma = hypo

            # compute log likelihood of m, given hypo
            stderr_m = sigma / math.sqrt(n)
            loglike = evalGaussianLogPdf(m, mu, stderr_m)

            #compute log likelihood of s, given hypo
            stderr_s = sigma / math.sqrt(2 * (n-1))
            loglike += evalGaussianLogPdf(s, sigma, stderr_s)

            self.incr(hypo, loglike)


def evalGaussianLogPdf(x, mu, sigma):
    """Computes the log PDF of x given mu and sigma.

    x: float values
    mu, sigma: paramemters of Gaussian

    returns: float log-likelihood
    """
    return scipy.stats.norm.logpdf(x, mu, sigma)


def findPriorRanges(xs, num_points, num_stderrs=3.0, median_flag=False):
    """Find ranges for mu and sigma with non-negligible likelihood.

    xs: sample
    num_points: number of values in each dimension
    num_stderrs: number of standard errors to include on either side
    
    Returns: sequence of mus, sequence of sigmas    
    """
    def makeRange(estimate, stderr):
        """Makes a linear range around the estimate.

        estimate: central value
        stderr: standard error of the estimate

        returns: numpy array of float
        """
        spread = stderr * num_stderrs
        array = numpy.linspace(estimate-spread, estimate+spread, num_points)
        return array

    # estimate mean and stddev of xs
    n = len(xs)
    if median_flag:
        m, s = medianS(xs, num_sigmas=NUM_SIGMAS)
    else:
        m = numpy.mean(xs)
        s = numpy.std(xs)

    print('classical estimators', m, s)

    # compute ranges for m and s
    stderr_m = s / math.sqrt(n)
    mus = makeRange(m, stderr_m)

    stderr_s = s / math.sqrt(2 * (n-1))
    sigmas = makeRange(s, stderr_s)

    return mus, sigmas


def summation(xs, mu, cache={}):
    """Computes the sum of (x-mu)**2 for x in t.

    Caches previous results.

    xs: tuple of values
    mu: hypothetical mean
    cache: cache of previous results
    """
    try:
        return cache[xs, mu]
    except KeyError:
        ds = [(x-mu)**2 for x in xs]
        total = sum(ds)
        cache[xs, mu] = total
        return total


def coefVariation(suite):
    """Computes the distribution of CV.

    suite: Pmf that maps (x, y) to z

    Returns: Pmf object for CV.
    """
    pmf = thinkbayes.PMF()
    for (m, s), p in suite.items():
        pmf.incr(s / m, p)
    return pmf


def plotCdfs(d, labels):
    """Plot CDFs for each sequence in a dictionary.

    Jitters the data and subtracts away the mean.

    d: map from key to sequence of values
    labels: map from key to string label
    """
    thinkplot.clf()
    for key, xs in d.iteritems():
        mu = thinkstats.mean(xs)
        xs = thinkstats.jitter(xs, 1.3)
        xs = [x-mu for x in xs]
        cdf = thinkbayes.makeCdfFromList(xs)
        thinkplot.cdf(cdf, label=labels[key])
    thinkplot.show()
                  

def plotPosterior(suite, pcolor=False, contour=True):
    """Makes a contour plot.
    
    suite: Suite that maps (mu, sigma) to probability
    """
    thinkplot.clf()
    thinkplot.contour(suite.getDict(), pcolor=pcolor, contour=contour)

    thinkplot.save(root='variability_posterior_%s' % suite.name,
                   title='Posterior joint distribution',
                   xlabel='Mean height (cm)',
                   ylabel='Stddev (cm)')


def plotCoefVariation(suites):
    """Plot the posterior distributions for CV.

    suites: map from label to Pmf of CVs.
    """
    thinkplot.clf()
    thinkplot.prePlot(num=2)

    pmfs = {}
    for label, suite in suites.iteritems():
        pmf = coefVariation(suite)
        print('CV posterior mean', pmf.mean())
        cdf = thinkbayes.makeCdfFromPmf(pmf, label)
        thinkplot.cdf(cdf)
    
        pmfs[label] = pmf

    thinkplot.save(root='variability_cv',
                   xlabel='Coefficient of variation',
                   ylabel='Probability')

    print('female bigger', thinkbayes.pmfProbGreater(pmfs['female'],
                                                     pmfs['male']))
    print('male bigger', thinkbayes.pmfProbGreater(pmfs['male'],
                                                   pmfs['female']))


def plotOutliers(samples):
    """Make CDFs showing the distribution of outliers."""
    cdfs = []
    for label, sample in samples.iteritems():
        outliers = [x for x in sample if x < 150]

        cdf = thinkbayes.makeCdfFromList(outliers, label)
        cdfs.append(cdf)

    thinkplot.clf()
    thinkplot.cdfs(cdfs)
    thinkplot.save(root='variability_cdfs',
                   title='CDF of height',
                   xlabel='Reported height (cm)',
                   ylabel='CDF')


def plotMarginals(suite):
    """Plots marginal distributions from a joint distribution.

    suite: joint distribution of mu and sigma.
    """
    thinkplot.clf()

    pyplot.subplot(1, 2, 1)
    pmf_m = suite.marginal(0)
    cdf_m = thinkbayes.makeCdfFromPmf(pmf_m)
    thinkplot.cdf(cdf_m)

    pyplot.subplot(1, 2, 2)
    pmf_s = suite.marginal(1)
    cdf_s = thinkbayes.makeCdfFromPmf(pmf_s)
    thinkplot.cdf(cdf_s)

    thinkplot.show()


def dumpHeights(data_dir='.', n=10000):
    """Read the BRFSS dataset, extract the heights and pickle them."""
    resp = brfss.Respondents()
    resp.readRecords(data_dir, n)

    d = {1:[], 2:[]}
    [d[r.sex].append(r.htm3) for r in resp.records if r.htm3 != 'NA']

    fp = open('variability_data.pkl', 'wb')
    pickle.dump(d, fp)
    fp.close()


def loadHeights():
    """Read the pickled height data.

    returns: map from sex code to list of heights.
    """
    fp = open('variability_data.pkl', 'r')
    d = pickle.load(fp)
    fp.close()
    return d


def updateSuite1(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes untransformed likelihoods.

    suite: Suite that maps from (mu, sigma) to prob
    xs: sequence
    """
    suite.updateSet(xs)


def updateSuite2(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods.

    suite: Suite that maps from (mu, sigma) to prob
    xs: sequence
    """
    suite.log()
    suite.logUpdateSet(xs)
    suite.exp()
    suite.normalize()


def updateSuite3(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.log()
    suite.logUpdateSetFast(xs)
    suite.exp()
    suite.normalize()


def updateSuite4(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.log()
    suite.logUpdateSetMeanVar(xs)
    suite.exp()
    suite.normalize()


def updateSuite5(suite, xs):
    """Computes the posterior distibution of mu and sigma.

    Computes log likelihoods efficiently.

    suite: Suite that maps from (mu, sigma) to prob
    t: sequence
    """
    suite.log()
    suite.logUpdateSetMedianIPR(xs)
    suite.exp()
    suite.normalize()


def medianIPR(xs, p):
    """Computes the median and interpercentile range.

    xs: sequence of values
    p: range (0-1), 0.5 yields the interquartile range

    returns: tuple of float (median, IPR)
    """
    cdf = thinkbayes.makeCdfFromList(xs)
    median = cdf.percentile(50)

    alpha = (1-p) / 2
    ipr = cdf.value(1 - alpha) - cdf.value(alpha)
    return median, ipr


def medianS(xs, num_sigmas):
    """Computes the median and an estimate of sigma.

    Based on an interpercentile range (IPR).

    factor: number of standard deviations spanned by the IPR
    """
    half_p = thinkbayes.standardGaussianCdf(num_sigmas) - 0.5
    median, ipr = medianIPR(xs, half_p * 2)
    s = ipr / 2 / num_sigmas

    return median, s

def summarize(xs):
    """Prints summary statistics from a sequence of values.

    xs: sequence of values
    """
    # print smallest and largest
    xs.sort()
    print('smallest', xs[:10])
    print('largest', xs[-10:])

    # print median and interquartile range
    cdf = thinkbayes.makeCdfFromList(xs)
    print(cdf.percentile(25), cdf.percentile(50), cdf.percentile(75))


def runEstimate(update_func, num_points=31, median_flag=False):
    """Runs the whole analysis.

    update_func: which of the update functions to use
    num_points: number of points in the Suite (in each dimension)
    """
    # DumpHeights(n=10000000)
    d = loadHeights()
    labels = {1:'male', 2:'female'}

    # PlotCdfs(d, labels)

    suites = {}
    for key, xs in d.iteritems():
        name = labels[key]
        print(name, len(xs))
        summarize(xs)

        xs = thinkstats.jitter(xs, 1.3)

        mus, sigmas = findPriorRanges(xs, num_points, median_flag=median_flag)
        suite = Height(mus, sigmas, name)
        suites[name] = suite
        update_func(suite, xs)
        print('MLE', suite.maximumLikelihood())

        plotPosterior(suite)

        pmf_m = suite.marginal(0)
        pmf_s = suite.marginal(1)
        print('marginal mu', pmf_m.mean(), pmf_m.variance())
        print('marginal sigma', pmf_s.mean(), pmf_s.variance())

        # PlotMarginals(suite)

    plotCoefVariation(suites)


def main():
    random.seed(17)

    func = updateSuite5
    median_flag = (func == updateSuite5)
    runEstimate(func, median_flag=median_flag)


if __name__ == '__main__':
    main()


""" Results:

UpdateSuite1 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite2 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite3 (100):
marginal mu 162.816901408 0.55779791443
marginal sigma 6.36966103214 0.277026082819

UpdateSuite4 (100):
marginal mu 162.816901408 0.547456009605
marginal sigma 6.30305516111 0.27544106054

UpdateSuite3 (1000):
marginal mu 163.722137405 0.0660294386397
marginal sigma 6.64453251495 0.0329935312671

UpdateSuite4 (1000):
marginal mu 163.722137405 0.0658920503302
marginal sigma 6.63692197049 0.0329689887609

UpdateSuite3 (all):
marginal mu 163.223475005 0.000203282582659
marginal sigma 7.26918836916 0.000101641131229

UpdateSuite4 (all):
marginal mu 163.223475004 0.000203281499857
marginal sigma 7.26916693422 0.000101640932082

UpdateSuite5 (all):
marginal mu 163.1805214 7.9399898468e-07
marginal sigma 7.29969524118 3.26257030869e-14

"""

