"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkbayes

import thinkplot
import numpy

import math
import random
import sys




# FORMATS = ['pdf', 'eps', 'png', 'jpg']
FORMATS = ['pdf']

"""
Notation guide:

z: time between trains
x: time since the last train
y: time until the next train

zb: distribution of z as seen by a random arrival

"""

# longest hypothetical time between trains, in seconds

UPPER_BOUND = 1200

# observed gaps between trains, in seconds
# collected using code in redline_data.py, run daily 4-6pm
# for 5 days, Monday 6 May 2013 to Friday 10 May 2013

OBSERVED_GAP_TIMES = [
    428.0, 705.0, 407.0, 465.0, 433.0, 425.0, 204.0, 506.0, 143.0, 351.0, 
    450.0, 598.0, 464.0, 749.0, 341.0, 586.0, 754.0, 256.0, 378.0, 435.0, 
    176.0, 405.0, 360.0, 519.0, 648.0, 374.0, 483.0, 537.0, 578.0, 534.0, 
    577.0, 619.0, 538.0, 331.0, 186.0, 629.0, 193.0, 360.0, 660.0, 484.0, 
    512.0, 315.0, 457.0, 404.0, 740.0, 388.0, 357.0, 485.0, 567.0, 160.0, 
    428.0, 387.0, 901.0, 187.0, 622.0, 616.0, 585.0, 474.0, 442.0, 499.0, 
    437.0, 620.0, 351.0, 286.0, 373.0, 232.0, 393.0, 745.0, 636.0, 758.0,
]


def biasPmf(pmf, name='', invert=False):
    """Returns the Pmf with oversampling proportional to value.

    If pmf is the distribution of true values, the result is the
    distribution that would be seen if values are oversampled in
    proportion to their values; for example, if you ask students
    how big their classes are, large classes are oversampled in
    proportion to their size.

    If invert=True, computes in inverse operation; for example,
    unbiasing a sample collected from students.

    Args:
      pmf: Pmf object.
      name: string name for the new Pmf.
      invert: boolean

     Returns:
       Pmf object
    """
    new_pmf = pmf.copy(name=name)

    for x in pmf.keys():
        if invert:
            new_pmf.mult(x, 1.0 / x)
        else:
            new_pmf.mult(x, x)
        
    new_pmf.normalize()
    return new_pmf


def unbiasPmf(pmf, name=''):
    """Returns the Pmf with oversampling proportional to 1/value.

    Args:
      pmf: Pmf object.
      name: string name for the new Pmf.

     Returns:
       Pmf object
    """
    return biasPmf(pmf, name, invert=True)


def makeUniformPmf(low, high):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusive)
    """
    pmf = thinkbayes.PMF()
    for x in makeRange(low=low, high=high):
        pmf.set(x, 1)
    pmf.normalize()
    return pmf    
    

def makeRange(low=10, high=None, skip=10):
    """Makes a range representing possible gap times in seconds.

    low: where to start
    high: where to end
    skip: how many to skip
    """
    if high is None:
        high = UPPER_BOUND

    return range(low, high+skip, skip)


class WaitTimeCalculator(object):
    """Encapsulates the forward inference process.

    Given the actual distribution of gap times (z),
    computes the distribution of gaps as seen by
    a random passenger (zb), which yields the distribution
    of wait times (y) and the distribution of elapsed times (x).
    """

    def __init__(self, pmf, inverse=False):
        """Constructor.

        pmf: Pmf of either z or zb
        inverse: boolean, true if pmf is zb, false if pmf is z
        """
        if inverse:
            self.pmf_zb = pmf
            self.pmf_z = unbiasPmf(pmf, name="z")
        else:
            # NOTE: pmf_z is unbiased dist, pmf_zb is biased, where passengers arrive in busy interval
            self.pmf_z = pmf
            self.pmf_zb = biasPmf(pmf, name="zb")

        # distribution of wait time
        self.pmf_y = pmfOfWaitTime(self.pmf_zb)

        # the distribution of elapsed time is the same as the
        # distribution of wait time
        self.pmf_x = self.pmf_y

    def generateSampleWaitTimes(self, n):
        """Generates a random sample of wait times.

        n: sample size

        Returns: sequence of values
        """
        cdf_y = thinkbayes.makeCdfFromPmf(self.pmf_y)
        sample = cdf_y.sample(n)
        return sample

    def generateSampleGaps(self, n):
        """Generates a random sample of gaps seen by passengers.

        n: sample size

        Returns: sequence of values
        """
        cdf_zb = thinkbayes.makeCdfFromPmf(self.pmf_zb)
        sample = cdf_zb.sample(n)
        return sample

    def generateSamplePassengers(self, lmbda, n):
        """Generates a sample wait time and number of arrivals.

        lam: arrival rate in passengers per second
        n: number of samples

        Returns: list of (k1, y, k2) tuples
        k1: passengers there on arrival
        y: wait time
        k2: passengers arrived while waiting
        """
        zs = self.generateSampleGaps(n)
        xs, ys = splitGaps(zs)

        res = []
        for x, y in zip(xs, ys):
            k1 = numpy.random.poisson(lmbda * x)
            k2 = numpy.random.poisson(lmbda * y)
            res.append((k1, y, k2))

        return res

    def plotPmfs(self, root='redline0'):
        """Plots the computed Pmfs.

        root: string
        """
        pmfs = scaleDists([self.pmf_z, self.pmf_zb], 1.0 / 60)

        thinkplot.clf()
        thinkplot.prePlot(2)
        thinkplot.pmfs(pmfs)
        thinkplot.save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


    def makePlot(self, root='redline2'):
        """Plots the computed CDFs.

        root: string
        """
        print('Mean z', self.pmf_z.mean() / 60)
        print('Mean zb', self.pmf_zb.mean() / 60)
        print('Mean y', self.pmf_y.mean() / 60)

        cdf_z = self.pmf_z.makeCdf()
        cdf_zb = self.pmf_zb.makeCdf()
        cdf_y = self.pmf_y.makeCdf()

        cdfs = scaleDists([cdf_z, cdf_zb, cdf_y], 1.0 / 60)

        thinkplot.clf()
        thinkplot.prePlot(3)
        thinkplot.cdfs(cdfs)
        thinkplot.save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


def splitGaps(zs):
    """Splits zs into xs and ys.

    zs: sequence of gaps

    Returns: tuple of sequences (xs, ys)
    """
    xs = [random.uniform(0, z) for z in zs]
    ys = [z-x for z, x in zip(zs, xs)]
    return xs, ys


def pmfOfWaitTime(pmf_zb):
    """Distribution of wait time.

    pmf_zb: dist of gap time as seen by a random observer (the biased distribution)

    Returns: dist of wait time (also dist of elapsed time)
    """
    # NOTE: zb = Y + X, where Y = dist of wait time, and X = dist of elapsed time.
    # NOTE: Y = wait times = time between arrival of passenger and arrival of train.
    # NOTE: X = wait time = time between arrival of previous train and arrival of next passenger.
    metaPmf = thinkbayes.PMF()
    for gap, prob in pmf_zb.items():
        uniform = makeUniformPmf(0, gap)
        metaPmf.set(uniform, prob)

    pmf_y = thinkbayes.makeMixture(metaPmf, name='y')
    return pmf_y


def scaleDists(dists, factor):
    """Scales each of the distributions in a sequence.

    dists: sequence of Pmf or Cdf
    factor: float scale factor
    """
    return [dist.scale(factor) for dist in dists]


class ElapsedTimeEstimator(object):
    """Uses the number of passengers to estimate time since last train."""

    def __init__(self, wtc, lmbda, num_passengers):
        """Constructor.

        pmf_x: expected distribution of elapsed time
        lam: arrival rate in passengers per second
        num_passengers: # passengers seen on the platform
        """
        # prior for elapsed time
        self.prior_x = Elapsed(wtc.pmf_x, name='prior x')

        # posterior of elapsed time (based on number of passengers)
        self.post_x = self.prior_x.copy(name='posterior x')
        self.post_x.update((lmbda, num_passengers))

        # predictive distribution of wait time
        self.pmf_y = predictWaitTime(wtc.pmf_zb, self.post_x)

    def makePlot(self, root='redline3'):
        """Plot the CDFs.

        root: string
        """
        # observed gaps
        cdf_prior_x = self.prior_x.makeCdf()
        cdf_post_x = self.post_x.makeCdf()
        cdf_y = self.pmf_y.makeCdf()

        cdfs = scaleDists([cdf_prior_x, cdf_post_x, cdf_y], 1.0 / 60)

        thinkplot.clf()
        thinkplot.prePlot(3)
        thinkplot.cdfs(cdfs)
        thinkplot.save(root=root,
                       xlabel='Time (min)',
                       ylabel='CDF',
                       formats=FORMATS)


class ArrivalRate(thinkbayes.Suite):
    """Represents the distribution of arrival rates (lambda)."""

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: arrival rate in passengers per second
        data: tuple of elapsed_time and number of passengers
        """
        lmbda = hypo
        x, k = data
        like = thinkbayes.evalPoissonPmf(k, lmbda * x)
        return like


class ArrivalRateEstimator(object):
    """Estimates arrival rate based on passengers that arrive while waiting.
    """

    def __init__(self, passenger_data):
        """Constructor

        passenger_data: sequence of (k1, y, k2) pairs
        """
        # range for lambda
        low, high = 0, 5
        n = 51
        hypos = numpy.linspace(low, high, n) / 60 # make hypothetical hypo rate lambda values.

        self.priorLambda = ArrivalRate(hypos, name='prior')
        self.priorLambda.remove(0)

        self.posteriorLambda = self.priorLambda.copy(name='posterior')

        for _k1, y, k2 in passenger_data:
            self.posteriorLambda.update((y, k2))

        print('Mean posterior lambda', self.posteriorLambda.mean())

    def makePlot(self, root='redline1'):
        """Plot the prior and posterior CDF of passengers arrival rate.

        root: string
        """
        thinkplot.clf()
        thinkplot.prePlot(2)

        # convert units to passengers per minute
        prior = self.priorLambda.makeCdf().scale(60)
        post = self.posteriorLambda.makeCdf().scale(60)

        thinkplot.cdfs([prior, post])

        thinkplot.save(root=root,
                       xlabel='Arrival rate (passengers / min)',
                       ylabel='CDF',
                       formats=FORMATS)
                       

class Elapsed(thinkbayes.Suite):
    """Represents the distribution of elapsed time (x)."""

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: elapsed time since the last train
        data: tuple of arrival rate and number of passengers
        """
        x = hypo
        lmbda, k = data
        like = thinkbayes.evalPoissonPmf(k, lmbda * x)
        return like


def predictWaitTime(pmf_zb, pmf_x):
    """Computes the distribution of wait times.

    Enumerate all pairs of zb from pmf_zb and x from pmf_x,
    and accumulate the distribution of y = z - x.

    pmf_zb: distribution of gaps seen by random observer
    pmf_x: distribution of elapsed time
    """
    pmf_y = pmf_zb - pmf_x
    pmf_y.name = 'pred y'
    removeNegatives(pmf_y) # impossible to wait more than 5 min if you are  in gap of 5 min
    return pmf_y


# fixed: made it list(pmf.values) instead of pmfkeys
def removeNegatives(pmf):
    """Removes negative values from a PMF.

    pmf: Pmf
    """
    for val in list(pmf.keys()):
        if val < 0:
            pmf.remove(val)
    pmf.normalize()



class Gaps(thinkbayes.Suite):
    """Represents the distribution of gap times,
    as updated by an observed waiting time."""

    def likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.

        If the actual gap time is z, what is the likelihood
        of waiting y seconds?

        hypo: actual time between trains
        data: observed wait time
        """
        z = hypo
        y = data
        if y > z:
            return 0
        return 1.0 / z


class GapDirichlet(thinkbayes.Dirichlet):
    """Represents the distribution of prevalences for each
    gap time."""

    def __init__(self, xs):
        """Constructor.

        xs: sequence of possible gap times
        """
        n = len(xs)
        thinkbayes.Dirichlet.__init__(self, n)
        self.xs = xs
        self.mean_zbs = []

    def pmfMeanZb(self):
        """Makes the Pmf of mean zb.

        Values stored in mean_zbs.
        """
        return thinkbayes.makePmfFromList(self.mean_zbs)

    def preload(self, data):
        """Adds pseudocounts to the parameters.

        data: sequence of pseudocounts
        """
        thinkbayes.Dirichlet.update(self, data)

    def update(self, data):
        """Computes the likelihood of the data.

        data: wait time observed by random arrival (y)

        Returns: float probability
        """
        k, y = data

        print(k, y)
        prior = self.predictivePmf(self.xs)
        gaps = Gaps(prior)
        gaps.update(y)
        probs = gaps.probs(self.xs)

        self.params += numpy.array(probs)


class GapDirichlet2(GapDirichlet):
    """Represents the distribution of prevalences for each
    gap time."""

    def update(self, data):
        """Computes the likelihood of the data.

        data: wait time observed by random arrival (y)

        Returns: float probability
        """
        k, y = data

        # get the current best guess for pmf_z
        pmf_zb = self.predictivePmf(self.xs)

        # use it to compute prior pmf_x, pmf_y, pmf_z
        wtc = WaitTimeCalculator(pmf_zb, inverse=True)

        # use the observed passengers to estimate posterior pmf_x
        elapsed = ElapsedTimeEstimator(wtc,
                                       lmbda=0.0333,
                                       num_passengers=k)

        # use posterior_x and observed y to estimate observed z
        obs_zb = elapsed.post_x + floor(y)
        probs = obs_zb.Probs(self.xs)

        mean_zb = obs_zb.Mean()
        self.mean_zbs.append(mean_zb)
        print(k, y, mean_zb)

        # use observed z to update beliefs about pmf_z
        self.params += numpy.array(probs)


class GapTimeEstimator(object):
    """Infers gap times using passenger data."""

    def __init__(self, xs, pcounts, passenger_data):
        self.xs = xs
        self.pcounts = pcounts
        self.passenger_data = passenger_data

        self.wait_times = [y for _k1, y, _k2 in passenger_data]
        self.pmf_y = thinkbayes.makePmfFromList(self.wait_times, name="y")

        dirichlet = GapDirichlet2(self.xs)
        dirichlet.params /= 1.0

        dirichlet.preload(self.pcounts)
        dirichlet.params /= 20.0

        self.prior_zb = dirichlet.predictivePmf(self.xs, name="prior zb")
        
        for k1, y, _k2 in passenger_data:
            dirichlet.update((k1, y))

        self.pmf_mean_zb = dirichlet.pmfMeanZb()

        self.post_zb = dirichlet.predictivePmf(self.xs, name="post zb")
        self.post_z = unbiasPmf(self.post_zb, name="post z")

    def plotPmfs(self):
        """Plot the PMFs."""
        print('Mean y', self.pmf_y.mean())
        print('Mean z', self.post_z.mean())
        print('Mean zb', self.post_zb.mean())

        thinkplot.pmf(self.pmf_y)
        thinkplot.pmf(self.post_z)
        thinkplot.pmf(self.post_zb)

    def makePlot(self):
        """Plot the CDFs."""
        thinkplot.cdf(self.pmf_y.makeCdf())
        thinkplot.cdf(self.prior_zb.makeCdf())
        thinkplot.cdf(self.post_zb.makeCdf())
        thinkplot.cdf(self.pmf_mean_zb.makeCdf())
        thinkplot.show()


def floor(x, factor=10):
    """Rounds down to the nearest multiple of factor.

    When factor=10, all numbers from 10 to 19 get floored to 10.
    """
    return int(x/factor) * factor


'''
def testGte():
    """Tests the GapTimeEstimator."""
    random.seed(17)

    xs = [60, 120, 240]

    gap_times = [60, 60, 60, 60, 60, 120, 120, 120, 240, 240]

    # distribution of gap time (z)
    pdf_z = thinkbayes.EstimatedPDF(gap_times)
    pmf_z = pdf_z.makePmf(xs, name="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)

    lam = 0.0333
    n = 100
    passenger_data = wtc.generateSamplePassengers(lam, n)

    pcounts = [0, 0, 0]

    ite = GapTimeEstimator(xs, pcounts, passenger_data)

    thinkplot.clf()

    # thinkplot.Cdf(wtc.pmf_z.MakeCdf(name="actual z"))
    thinkplot.cdf(wtc.pmf_zb.makeCdf(name="actual zb"))
    ite.makePlot()
'''


class WaitMixtureEstimator(object):
    """Encapsulates the process of estimating wait time with uncertain lam.
    """

    def __init__(self, wtc, are, num_passengers=15):
        """Constructor.

        wtc: WaitTimeCalculator
        are: ArrivalTimeEstimator
        num_passengers: number of passengers seen on the platform
        """
        self.metaPmf = thinkbayes.PMF()

        for lmbda, prob in sorted(are.posteriorLambda.items()):
            ete = ElapsedTimeEstimator(wtc, lmbda, num_passengers)
            self.metaPmf.set(ete.pmf_y, prob)

        self.mixture = thinkbayes.makeMixture(self.metaPmf)

        lmbda = are.posteriorLambda.mean()
        ete = ElapsedTimeEstimator(wtc, lmbda, num_passengers)
        self.point = ete.pmf_y

    def makePlot(self, root='redline4'):
        """Makes a plot showing the mixture."""
        thinkplot.clf()

        # plot the MetaPmf
        for pmf, prob in sorted(self.metaPmf.items()):
            cdf = pmf.makeCdf().scale(1.0 / 60)
            width = 2/math.log(-math.log(prob))
            thinkplot.plot(cdf.xs, cdf.ps,
                           alpha=0.2, linewidth=width, color='blue',
                           label='')

        # plot the mixture and the distribution based on a point estimate
        thinkplot.prePlot(2)
        #thinkplot.Cdf(self.point.MakeCdf(name='point').Scale(1.0/60))
        thinkplot.cdf(self.mixture.makeCdf(name='mix').scale(1.0 / 60))

        thinkplot.save(root=root,
                       xlabel='Wait time (min)',
                       ylabel='CDF',
                       formats=FORMATS,
                       axis=[0,10,0,1])



def generateSampleData(gap_times, lmbda=0.0333, n=10):
    """Generates passenger data based on actual gap times.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    n: number of simulated observations
    """
    xs = makeRange(low=10)
    pdf_z = thinkbayes.EstimatedPDF(gap_times)
    pmf_z = pdf_z.makePmf(xs, name="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)
    passenger_data = wtc.generateSamplePassengers(lmbda, n)
    return wtc, passenger_data


def randomSeed(x):
    """Initialize the random and numpy.random generators.

    x: int seed
    """
    random.seed(x)
    numpy.random.seed(x)
    

def runSimpleProcess(gap_times, lmbda=0.0333, num_passengers=15, plot=True):
    """Runs the basic analysis and generates figures.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    num_passengers: int number of passengers on the platform
    plot: boolean, whether to generate plots

    Returns: WaitTimeCalculator, ElapsedTimeEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 1200

    cdf_z = thinkbayes.makeCdfFromList(gap_times).scale(1.0 / 60)
    print('CI z', cdf_z.credibleInterval(90))

    xs = makeRange(low=10)

    pdf_z = thinkbayes.EstimatedPDF(gap_times)
    pmf_z = pdf_z.makePmf(xs, name="z")

    wtc = WaitTimeCalculator(pmf_z, inverse=False)    

    if plot:
        wtc.plotPmfs()
        wtc.makePlot()

    ete = ElapsedTimeEstimator(wtc, lmbda, num_passengers)

    if plot:
        ete.makePlot()

    return wtc, ete


def runMixProcess(gap_times, lmbda=0.0333, num_passengers=15, plot=True):
    """Runs the analysis for unknown lambda.

    gap_times: sequence of float
    lam: arrival rate in passengers per second
    num_passengers: int number of passengers on the platform
    plot: boolean, whether to generate plots

    Returns: WaitMixtureEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 1200

    wtc, _ete = runSimpleProcess(gap_times, lmbda, num_passengers)

    randomSeed(20)
    passenger_data = wtc.generateSamplePassengers(lmbda, n=5)

    total_y = 0
    total_k2 = 0
    for k1, y, k2 in passenger_data:
        print(k1, y/60, k2)
        total_y += y/60
        total_k2 += k2
    print(total_k2, total_y)
    print('Average arrival rate', total_k2 / total_y)

    # NOTE: makes figure redline1 - mean and median of posterior are near 2 psgrs/min but has wide spread because of little data.
    are = ArrivalRateEstimator(passenger_data)

    if plot:
        are.makePlot()

    wme = WaitMixtureEstimator(wtc, are, num_passengers)

    if plot:
        wme.makePlot()

    return wme


def runLoop(gap_times, nums, lmbda=0.0333):
    """Runs the basic analysis for a range of num_passengers.

    gap_times: sequence of float
    nums: sequence of values for num_passengers
    lam: arrival rate in passengers per second

    Returns: WaitMixtureEstimator
    """
    global UPPER_BOUND
    UPPER_BOUND = 4000

    thinkplot.clf()

    randomSeed(18)

    # resample gap_times
    n = 220
    cdf_z = thinkbayes.makeCdfFromList(gap_times)
    sample_z = cdf_z.sample(n)
    pmf_z = thinkbayes.makePmfFromList(sample_z)

    # compute the biased pmf and add some long delays
    cdf_zp = biasPmf(pmf_z).makeCdf()
    sample_zb = cdf_zp.sample(n) + [1800, 2400, 3000]

    # smooth the distribution of zb
    pdf_zb = thinkbayes.EstimatedPDF(sample_zb)
    xs = makeRange(low=60)
    pmf_zb = pdf_zb.makePmf(xs)

    # unbias the distribution of zb and make wtc
    pmf_z = unbiasPmf(pmf_zb)
    wtc = WaitTimeCalculator(pmf_z)



    # NOTE: THis is the prob of long wait part on page 89
    # Given number of passengers on platform, problongwait makes an
    # * elapsedtimeestimator
    # * extracts dist of wait time (y)
    # * compute probability that wait time exceeds minutes (15 here)
    # RESULT PLOT: when passgrs num < 20, system isoperating normally so prob of long delay is small
    # But if greater than 30 pssgrs, then it has been 15 mins since last train, which is longer than
    # normal delay so need to take taxi.
    probs = []
    for num_passengers in nums:
        ete = ElapsedTimeEstimator(wtc, lmbda, num_passengers)

        # compute the posterior prob of waiting more than 15 minutes
        cdf_y = ete.pmf_y.makeCdf()
        prob = 1 - cdf_y.prob(900)
        probs.append(prob)

        # thinkplot.Cdf(ete.pmf_y.MakeCdf(name=str(num_passengers)))
    
    thinkplot.plot(nums, probs)
    thinkplot.save(root='redline5',
                   xlabel='Num passengers',
                   ylabel='P(y > 15 min)',
                   formats=FORMATS,
                   )


def main(script):
    runLoop(OBSERVED_GAP_TIMES, nums=[0, 5, 10, 15, 20, 25, 30, 35])
    runMixProcess(OBSERVED_GAP_TIMES)
    

if __name__ == '__main__':
    main(*sys.argv)
