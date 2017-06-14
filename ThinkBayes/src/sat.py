"""This file contains code used in "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import csv
import math
import sys

import numpy
import thinkbayes

from src import thinkplot


def readScale(filename='sat_scale.csv', col=2):
    """Reads a CSV file of SAT scales (maps from raw score to standard score).

    Args:
      filename: string filename
      col: which column to start with (0=Reading, 2=Math, 4=Writing)

    Returns: thinkbayes.Interpolator object
    """
    def parseRange(s):
        """Parse a range of values in the form 123-456

        s: string
        """
        t = [int(x) for x in s.split('-')]
        return 1.0 * sum(t) / len(t)

    fp = open(filename)
    reader = csv.reader(fp)
    raws = []
    scores = []

    for t in reader:
        try:
            raw = int(t[col])
            raws.append(raw)
            score = parseRange(t[col+1])
            scores.append(score)
        except ValueError:
            pass

    raws.sort()
    scores.sort()
    return thinkbayes.Interpolator(raws, scores)


def readRanks(filename='sat_ranks.csv'):
    """Reads a CSV file of SAT scores.

    Args:
      filename: string filename

    Returns:
      list of (score, freq) pairs
    """
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        try:
            score = int(t[0])
            freq = int(t[1])
            res.append((score, freq))
        except ValueError:
            pass

    return res


def divideValues(pmf, denom):
    """Divides the values in a Pmf by denom.

    Returns a new Pmf.
    """
    new = thinkbayes.PMF()
    denom = float(denom)
    for val, prob in pmf.items():
        x = val / denom
        new.set(x, prob)
    return new


class Exam(object):
    """Encapsulates information about an exam.

    Contains the distribution of scaled scores and an
    Interpolator that maps between scaled and raw scores.
    """
    def __init__(self):
        self.scale = readScale() # scale is interpolator (linear mapping) between raws and scores sorted.

        scores = readRanks()
        score_pmf = thinkbayes.makePmfFromDict(dict(scores)) # pmf of scaled scores

        self.raw = self.reverseScale(score_pmf) # pmf of raw scores (gets raw from scores and assigns probs)
        self.max_score = max(self.raw.keys()) # max raw score
        self.prior = divideValues(self.raw, denom=self.max_score) # pmf of p_correct (raw/highestraw, prob)

        center = -0.05
        width = 1.8
        self.difficulties = makeDifficulties(center, width, self.max_score)

    def compareScores(self, a_score, b_score, constructor):
        """Computes posteriors for two test scores and the likelihood ratio.

        a_score, b_score: scales SAT scores
        constructor: function that instantiates an Sat or Sat2 object
        """
        a_sat = constructor(self, a_score)
        b_sat = constructor(self, b_score)

        a_sat.plotPosteriors(b_sat)

        if constructor is Sat:
            plotJointDist(a_sat, b_sat)

        top = TopLevel('AB')
        top.update((a_sat, b_sat))

        print("Printing toplevel suite: ")
        top.printSuite() ## note: p(A) = 77% and p(B) = 23% ... Alice posterior is 77% (prob of efficacy)

        ratio = top.prob('A') / top.prob('B')

        print('Likelihood ratio = ', ratio) # =3.8 so Alice is better than Bob at SAT.

        posterior = ratio / (ratio + 1)
        print('Posterior = ', posterior)

        if constructor is Sat2:
            comparePosteriorPredictive(a_sat, b_sat)

    def makeRawScoreDist(self, efficacies):
        """Makes the distribution of raw scores for given difficulty.

        efficacies: Pmf of efficacy
        """
        pmfs = thinkbayes.PMF()
        for efficacy, prob in efficacies.items():
            scores = self.pmfCorrect(efficacy)
            pmfs.set(scores, prob)

        mix = thinkbayes.makeMixture(pmfs)
        return mix

    def calibrateDifficulty(self):
        """Make a plot showing the model distribution of raw scores."""
        thinkplot.clf()
        thinkplot.prePlot(num=2)

        cdf = thinkbayes.makeCdfFromPmf(self.raw, name='data')
        thinkplot.cdf(cdf)

        efficacies = thinkbayes.makeGaussianPmf(0, 1.5, 3)
        pmf = self.makeRawScoreDist(efficacies) # mixture model of raw score, prob = p1 * p2
        cdf = thinkbayes.makeCdfFromPmf(pmf, name='model')
        thinkplot.cdf(cdf)

        thinkplot.save(root='sat_2_calibrate',
                       xlabel='raw score',
                       ylabel='CDF',
                       formats=['pdf'])

    def pmfCorrect(self, efficacy):
        """Returns the PMF of number of correct responses.

        efficacy: float
        """
        pmf = pmfCorrect(efficacy, self.difficulties)
        return pmf

    def lookup(self, raw):
        """Looks up a raw score and returns a scaled score."""
        return self.scale.lookup(raw)

    def reverse(self, score):
        """Looks up a scaled score and returns a raw score.

        Since we ignore the penalty, negative scores round up to zero.
        """
        raw = self.scale.reverse(score)
        return raw if raw > 0 else 0

    def reverseScale(self, pmf):
        """Applies the reverse scale to the values of a PMF.

        Args:
            pmf: Pmf object
            scale: Interpolator object

        Returns:
            new Pmf
        """
        new = thinkbayes.PMF()
        for val, prob in pmf.items():
            raw = self.reverse(val)
            new.incr(raw, prob)
        return new


class Sat(thinkbayes.Suite):
    """Represents the distribution of p_correct for a test-taker."""

    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the prior distribution
        thinkbayes.Suite.__init__(self, exam.prior)

        # note: update based on an exam score
        self.update(score)



    def likelihood(self, data, hypo):
        """Computes the likelihood of a test score, given efficacy."""
        p_correct = hypo
        score = data

        k = self.exam.reverse(score)
        n = self.exam.max_score
        like = thinkbayes.evalBinomialPmf(k, n, p_correct)
        return like

    def plotPosteriors(self, other):
        """Plots posterior distributions of efficacy.

        self, other: Sat objects.
        """
        thinkplot.clf()
        thinkplot.prePlot(num=2)

        cdf1 = thinkbayes.makeCdfFromPmf(self, 'posterior %d' % self.score)
        cdf2 = thinkbayes.makeCdfFromPmf(other, 'posterior %d' % other.score)

        thinkplot.cdfs([cdf1, cdf2])
        thinkplot.save(xlabel='p_correct',
                       ylabel='CDF',
                       axis=[0.7, 1.0, 0.0, 1.0],
                       root='sat_3_posteriors_p_corr',
                       formats=['pdf'])


class Sat2(thinkbayes.Suite):
    """Represents the distribution of efficacy for a test-taker."""

    def __init__(self, exam, score):
        self.exam = exam
        self.score = score

        # start with the Gaussian prior
        efficacies = thinkbayes.makeGaussianPmf(0, 1.5, 3)
        thinkbayes.Suite.__init__(self, efficacies)

        # update based on an exam score
        self.update(score)

    def likelihood(self, data, hypo):
        """Computes the likelihood of a test score, given efficacy."""
        efficacy = hypo
        score = data
        raw = self.exam.reverse(score)

        pmf = self.exam.pmfCorrect(efficacy)
        like = pmf.prob(raw)
        return like

    def makePredictiveDist(self):
        """Returns the distribution of raw scores expected on a re-test."""
        raw_pmf = self.exam.makeRawScoreDist(self)
        return raw_pmf

    def plotPosteriors(self, other):
        """Plots posterior distributions of efficacy.

        self, other: Sat objects.
        """
        thinkplot.clf()
        thinkplot.prePlot(num=2)

        cdf1 = thinkbayes.makeCdfFromPmf(self, 'posterior %d' % self.score)
        cdf2 = thinkbayes.makeCdfFromPmf(other, 'posterior %d' % other.score)

        thinkplot.cdfs([cdf1, cdf2])
        thinkplot.save(xlabel='efficacy',
                       ylabel='CDF',
                       axis=[0, 4.6, 0.0, 1.0],
                       root='sat_5_posteriors_eff',
                       formats=['pdf'])


def plotJointDist(pmf1, pmf2, thresh=0.8):
    """Plot the joint distribution of p_correct.

    pmf1, pmf2: posterior distributions
    thresh: lower bound of the range to be plotted
    """
    def clean(pmf):
        """Removes values below thresh."""
        vals = [val for val in pmf.keys() if val < thresh]
        [pmf.remove(val) for val in vals]

    clean(pmf1)
    clean(pmf2)
    pmf = thinkbayes.makeJoint(pmf1, pmf2)

    thinkplot.figure(figsize=(6, 6))
    thinkplot.contour(pmf, contour=False, pcolor=True)

    thinkplot.plot([thresh, 1.0], [thresh, 1.0],
                   color='gray', alpha=0.2, linewidth=4)

    thinkplot.save(root='sat_4_joint',
                   xlabel='p_correct Alice',
                   ylabel='p_correct Bob',
                   axis=[thresh, 1.0, thresh, 1.0],
                   formats=['pdf'])


def comparePosteriorPredictive(a_sat, b_sat):
    """Compares the predictive distributions of raw scores.

    a_sat: posterior distribution
    b_sat:
    """
    a_pred = a_sat.makePredictiveDist()
    b_pred = b_sat.makePredictiveDist()

    #thinkplot.Clf()
    #thinkplot.Pmfs([a_pred, b_pred])
    #thinkplot.Show()

    a_like = thinkbayes.pmfProbGreater(a_pred, b_pred)  # pred prob Alice better
    b_like = thinkbayes.pmfProbLess(a_pred, b_pred) # pred prob Bob better.
    c_like = thinkbayes.pmfProbEqual(a_pred, b_pred) # pred prob a tie.

    print('Posterior predictive')
    print('A (pmf prob greater (a,b) = ', a_like) # 63% chance Alice does better on new test against Bob.
    print('B (pmf prob less (a,b) = ', b_like)
    print('C (pmf prob equal (a,b) = ', c_like)


def plotPriorDist(pmf):
    """Plot the prior distribution of p_correct.

    pmf: prior
    """
    thinkplot.clf()
    thinkplot.prePlot(num=1)

    cdf1 = thinkbayes.makeCdfFromPmf(pmf, 'prior')
    thinkplot.cdf(cdf1)
    thinkplot.save(root='sat_1_prior',
                   xlabel='p_correct',
                   ylabel='CDF',
                   formats=['pdf']) # ['pdf', 'eps'])


class TopLevel(thinkbayes.Suite):
    """Evaluates the top-level hypotheses about Alice and Bob.

    Uses the bottom-level posterior distribution about p_correct
    (or efficacy).
    """

    def update(self, data):
        a_sat, b_sat = data # posterior distributions are passed.

        a_like = thinkbayes.pmfProbGreater(a_sat, b_sat)
        b_like = thinkbayes.pmfProbLess(a_sat, b_sat)
        # round-off error beacuse p_correct is a discrete distribution
        c_like = thinkbayes.pmfProbEqual(a_sat, b_sat)

        # splitting this round off error evenly between a and b
        a_like += c_like / 2
        b_like += c_like / 2

        self.mult('A', a_like)
        self.mult('B', b_like)

        self.normalize()


def probCorrect(efficacy, difficulty, a=1):
    """Returns the probability that a person gets a question right.

    efficacy: personal ability to answer questions
    difficulty: how hard the question is

    Returns: float prob
    """
    return 1 / (1 + math.exp(-a * (efficacy - difficulty)))


def binaryPmf(p):
    """Makes a Pmf with values 1 and 0.

    p: probability given to 1

    Returns: Pmf object
    """
    pmf = thinkbayes.PMF()
    pmf.set(1, p)
    pmf.set(0, 1 - p)
    return pmf


def pmfCorrect(efficacy, difficulties):
    """Computes the distribution of correct responses.

    efficacy: personal ability to answer questions
    difficulties: list of difficulties, one for each question

    Returns: new Pmf object
    """
    pmf0 = thinkbayes.PMF([0])

    ps = [probCorrect(efficacy, difficulty) for difficulty in difficulties]
    pmfs = [binaryPmf(p) for p in ps]
    dist = sum(pmfs, pmf0)
    return dist


def makeDifficulties(center, width, n):
    """Makes a list of n difficulties with a given center and width.

    Returns: list of n floats between center-width and center+width
    """
    low, high = center-width, center+width
    return numpy.linspace(low, high, n)


def probCorrectTable():
    """Makes a table of p_correct for a range of efficacy and difficulty."""
    efficacies = [3, 1.5, 0, -1.5, -3]
    difficulties = [-1.85, -0.05, 1.75]

    for eff in efficacies:
        print('%0.2f & ' % eff)
        for diff in difficulties:
            p = probCorrect(eff, diff)
            print('%0.2f & ' % p)
        print(r'\\')

    # TODO don't know what this is supposed to mean
    '''
    for eff in efficacies:
        print '%0.2f & ' % eff,
        for diff in difficulties:
            p = probCorrect(eff, diff)
            print '%0.2f & ' % p, 
        print r'\\'
    '''


def main(script):
    print("\nPrinting prob correct table: ------------------------\n")
    probCorrectTable()

    exam = Exam()

    print("\nPlotting exam prior distribution: ------------------------")
    plotPriorDist(exam.prior)


    exam.calibrateDifficulty()

    print("\nComparing scores: ------------------------")
    exam.compareScores(780, 740, constructor=Sat)
    exam.compareScores(780, 740, constructor=Sat2)


if __name__ == '__main__':
    main(*sys.argv)
