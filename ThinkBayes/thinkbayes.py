"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

"""This file contains class definitions for:

Hist: represents a histogram (map from values to integer frequencies).

Pmf: represents a probability mass function (map from values to probs).

_DictWrapper: private parent class for Hist and Pmf.

Cdf: represents a discrete cumulative distribution function

Pdf: represents a continuous probability density function

"""

import bisect
import copy
import logging
import math
import numpy
import random
import pandas

import scipy.stats
from scipy.special import erf, erfinv

ROOT2 = math.sqrt(2)

def randomSeed(x):
    """Initialize the random and numpy.random generators.

    x: int seed
    """
    random.seed(x)
    numpy.random.seed(x)
    

def odds(p):
    """Computes odds for a given probability.

    Example: p=0.75 means 75 for and 25 against, or 3:1 odds in favor.

    Note: when p=1, the formula for odds divides by zero, which is
    normally undefined.  But I think it is reasonable to define Odds(1)
    to be infinity, so that's what this function does.

    p: float 0-1

    Returns: float odds
    """
    if p == 1:
        return float('inf')
    return p / (1 - p)


def probability(o):
    """Computes the probability corresponding to given odds.

    Example: o=2 means 2:1 odds in favor, or 2/3 probability

    o: float odds, strictly positive

    Returns: float probability
    """
    return o / (o + 1)


def probability2(yes, no):
    """Computes the probability corresponding to given odds.

    Example: yes=2, no=1 means 2:1 odds in favor, or 2/3 probability.
    
    yes, no: int or float odds in favor
    """
    return float(yes) / (yes + no)


class Interpolator(object):
    """Represents a mapping between sorted sequences; performs linear interp.

    Attributes:
        xs: sorted list
        ys: sorted list
    """

    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def lookup(self, x):
        """Looks up x and returns the corresponding value of y."""
        return self._bisect(x, self.xs, self.ys)

    def reverse(self, y):
        """Looks up y and returns the corresponding value of x."""
        return self._bisect(y, self.ys, self.xs)

    def _bisect(self, x, xs, ys):
        """Helper function."""
        if x <= xs[0]:
            return ys[0]
        if x >= xs[-1]:
            return ys[-1]
        i = bisect.bisect(xs, x)
        frac = 1.0 * (x - xs[i - 1]) / (xs[i] - xs[i - 1])
        y = ys[i - 1] + frac * 1.0 * (ys[i] - ys[i - 1])
        return y


class _DictWrapper(object):
    """An object that contains a dictionary."""

    def __init__(self, values=None, name=''):
        """Initializes the distribution.

        hypos: sequence of hypotheses
        """
        self.name = name
        self.d = {}

        # flag whether the distribution is under a log transform
        self.useLog = False

        if values is None:
            return

        init_methods = [
            self.initPmf,
            self.initMapping,
            self.initSequence,
            self.initFailure,
            ]

        for method in init_methods:
            try:
                method(values)
                break
            except AttributeError:
                continue

        if len(self) > 0:
            self.normalize()

    def initSequence(self, values):
        """Initializes with a sequence of equally-likely values.

        values: sequence of values
        """
        for value in values:
            self.set(value, 1)

    def initMapping(self, values):
        """Initializes with a map from value to probability.

        values: map from value to probability
        """
        for value, prob in values.iteritems(): #@ it's ok here to keep values.iteritems() not iter(values) -- seems to work only for VALUES not SELF.D!!! - probably because values = str list while self.d = dict
            self.set(value, prob)

    def initPmf(self, values):
        """Initializes with a Pmf.

        values: Pmf object
        """
        for value, prob in values.items():
            self.set(value, prob)

    def initFailure(self, values):
        """Raises an error."""
        raise ValueError('None of the initialization methods worked.')

    # fixed: added __hash__ and __eq__ here in dict but
    # fixed - removed __eq__ in PMF so it's hashabla so we can use metapmf
    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        try:
            return self.d == other.d
        except AttributeError:
            return False

    def __len__(self):
        return len(self.d)

    def __iter__(self):
        return iter(self.d)

    def iterkeys(self):
        return iter(self.d)

    def __contains__(self, value):
        return value in self.d

    def copy(self, name=None):
        """Returns a copy.

        Make a shallow copy of d.  If you want a deep copy of d,
        use copy.deepcopy on the whole object.

        Args:
            name: string name for the new Hist
        """
        new = copy.copy(self)
        new.d = copy.copy(self.d)
        new.name = name if name is not None else self.name
        return new

    def scale(self, factor):
        """Multiplies the values by a factor.

        factor: what to multiply by

        Returns: new object
        """
        new = self.copy()
        new.d.clear()

        for val, prob in self.items():
            new.set(val * factor, prob)
        return new

    def log(self, m=None):
        """Log transforms the probabilities.
        
        Removes values with probability 0.

        Normalizes so that the largest logprob is 0.
        """
        if self.useLog:
            raise ValueError("Pmf/Hist already under a log transform")
        self.useLog = True

        if m is None:
            m = self.maxLike()

        for x, p in iter(self.d):
            if p:
                self.set(x, math.log(p / m))
            else:
                self.remove(x)

    def exp(self, m=None):
        """Exponentiates the probabilities.

        m: how much to shift the ps before exponentiating

        If m is None, normalizes so that the largest prob is 1.
        """
        if not self.useLog:
            raise ValueError("Pmf/Hist not under a log transform")
        self.useLog = False

        if m is None:
            m = self.maxLike()

        for x, p in iter(self.d):
            self.set(x, math.exp(p - m))

    def getDict(self):
        """Gets the dictionary."""
        return self.d

    def setDict(self, d):
        """Sets the dictionary."""
        self.d = d

    def keys(self):
        """Gets an unsorted sequence of values.

        Note: one source of confusion is that the keys of this
        dictionary are the values of the Hist/Pmf, and the
        values of the dictionary are frequencies/probabilities.
        """
        return self.d.keys()

    def items(self):
        """Gets an unsorted sequence of (value, freq/prob) pairs."""
        return self.d.items()

    def render(self):
        """Generates a sequence of points suitable for plotting.

        Returns:
            tuple of (sorted value sequence, freq/prob sequence)
        """
        return zip(*sorted(self.items()))

    def print(self):
        """Prints the values and freqs/probs in ascending order."""
        for val, prob in sorted(iter(self.d.items())):
            print(val, ": ", prob, sep="")

    def set(self, x, y=0):
        """Sets the freq/prob associated with the value x.

        Args:
            x: number value
            y: number freq or prob
        """
        self.d[x] = y

    def incr(self, x, term=1):
        """Increments the freq/prob associated with the value x.

        Args:
            x: number value
            term: how much to increment by
        """
        self.d[x] = self.d.get(x, 0) + term

    def mult(self, x, factor):
        """Scales the freq/prob associated with the value x.

        Args:
            x: number value
            factor: how much to multiply by
        """
        self.d[x] = self.d.get(x, 0) * factor
        # MEANING: x = p(B1)/P(V) , factor = p(V|B1), where P(V) = normalizing constant

    def remove(self, x):
        """Removes a value.

        Throws an exception if the value is not there.

        Args:
            x: value to remove
        """
        del self.d[x]

    def total(self):
        """Returns the total of the frequencies/probabilities in the map."""
        total = sum(iter(self.d.values()))  #@fix iterate over values
        return total

    def maxLike(self):
        """Returns the largest frequency/probability in the map."""
        return max(iter(self.d.values())) #@fix iterate over values


class Hist(_DictWrapper):
    """Represents a histogram, which is a map from values to frequencies.

    Values can be any hashable type; frequencies are integer counters.
    """

    def freq(self, x):
        """Gets the frequency associated with the value x.

        Args:
            x: number value

        Returns:
            int frequency
        """
        return self.d.get(x, 0)

    def freqs(self, xs):
        """Gets frequencies for a sequence of values."""
        return [self.freq(x) for x in xs]

    def isSubset(self, other):
        """Checks whether the values in this histogram are a subset of
        the values in the given histogram."""
        for val, freq in self.items():
            if freq > other.freq(val):
                return False
        return True

    def subtract(self, other):
        """Subtracts the values in the given histogram from this histogram."""
        for val, freq in other.items():
            self.incr(val, -freq)


class PMF(_DictWrapper):
    """Represents a probability mass function.
    
    Values can be any hashable type; probabilities are floating-point.
    Pmfs are not necessarily normalized.
    """

    def prob(self, x, default=0):
        """Gets the probability associated with the value x.

        Args:
            x: number value
            default: value to return if the key is not there

        Returns:
            float probability
        """
        return self.d.get(x, default)

    def probs(self, xs):
        """Gets probabilities for a sequence of values."""
        return [self.prob(x) for x in xs]

    def makeCdf(self, name=None):
        """Makes a Cdf."""
        return makeCdfFromPmf(self, name=name)

    def probGreater(self, x):
        """Probability that a sample from this Pmf exceeds x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in iter(self.d) if val > x]
        return sum(t)

    def probLess(self, x):
        """Probability that a sample from this Pmf is less than x.

        x: number

        returns: float probability
        """
        t = [prob for (val, prob) in iter(self.d) if val < x]
        return sum(t)

    def __lt__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmfProbLess(self, obj)
        else:
            return self.probLess(obj)

    def __gt__(self, obj):
        """Greater than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmfProbGreater(self, obj)
        else:
            return self.probGreater(obj)

    def __ge__(self, obj):
        """Greater than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self < obj)

    def __le__(self, obj):
        """Less than or equal.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self > obj)

    # NOTE: erase __eq__ if you want PMF to be hashable, but put eq and hash in dict.
    '''
    def __eq__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        if isinstance(obj, _DictWrapper):
            return pmfProbEqual(self, obj)
        else:
            return self.prob(obj)
    '''


    def __ne__(self, obj):
        """Less than.

        obj: number or _DictWrapper

        returns: float probability
        """
        return 1 - (self == obj)


    def normalize(self, fraction=1.0):
        """Normalizes this PMF so the sum of all probs is fraction.

        Args:
            fraction: what the total should be after normalization

        Returns: the total probability before normalizing
        """
        #counter = 0
        if self.useLog:
            raise ValueError("Pmf is under a log transform")

        total = self.total()
        if total == 0.0:
            raise ValueError('total probability is zero.')
            logging.warning('Normalize: total probability is zero.')
            return total
        # turn 5/8 upside down to 8/5 to directly multiply with p(B1)p(V|B1)
        factor = float(fraction) / total
        for x in self.d:
            self.d[x] *= factor

        return total

    def random(self):
        """Chooses a random element from this PMF.

        Returns:
            float value from the Pmf
        """
        if len(self.d) == 0:
            raise ValueError('Pmf contains no values.')

        target = random.random()
        total = 0.0
        for x, p in iter(self.d.items()):   #@ fix added .items() to self.d
            total += p
            if total >= target:
                return x

        # we shouldn't get here
        assert False

    def mean(self): # changed by me
        """Computes the mean of a PMF.

        Returns:
            float mean
        """
        mu = 0.0
        # cast self.d to string so items can be iterable: TypeError: 'int' object is not iterable
        #dAsString = str((key, str(value)) for key, value in iter(self.d))
        for x, p in self.d.items():   #@fix iterate over values
            mu += p * x
        return mu

    def variance(self, mu=None):
        """Computes the variance of a PMF.

        Args:
            mu: the point around which the variance is computed;
                if omitted, computes the mean

        Returns:
            float variance
        """
        if mu is None:
            mu = self.mean()

        var = 0.0
        for x, p in iter(self.d):
            var += p * (x - mu) ** 2   # maybe here too?
        return var

    def maximumLikelihood(self):
        """Returns the value with the highest probability.

        Returns: float probability
        """
        prob, val = max((prob, val) for val, prob in self.items())
        return val

    def credibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        cdf = self.makeCdf()
        return cdf.credibleInterval(percentage)

    def __add__(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        try:
            return self.addPmf(other)
        except AttributeError:
            return self.addConstant(other)

    def addPmf(self, other):
        """Computes the Pmf of the sum of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = PMF()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 + v2, p1 * p2)
        return pmf

    def addConstant(self, other):
        """Computes the Pmf of the sum a constant and  values from self.

        other: a number

        returns: new Pmf
        """
        pmf = PMF()
        for v1, p1 in self.items():
            pmf.set(v1 + other, p1)   # MAYBE HERE?
        return pmf

    def __sub__(self, other):
        """Computes the Pmf of the diff of values drawn from self and other.

        other: another Pmf

        returns: new Pmf
        """
        pmf = PMF()
        for v1, p1 in self.items():
            for v2, p2 in other.items():
                pmf.incr(v1 - v2, p1 * p2)
        return pmf

    def max(self, k): # @ class Pmf
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.makeCdf()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


class Joint(PMF):
    """Represents a joint distribution.

    The values are sequences (usually tuples)
    """

    def marginal(self, i, name=''):
        """Gets the marginal distribution of the indicated variable.

        i: index of the variable we want

        Returns: Pmf
        """
        pmf = PMF(name=name)
        for vs, prob in self.items():
            pmf.incr(vs[i], prob)
        return pmf

    def conditional(self, i, j, val, name=''):
        """gets the conditional distribution of the indicated variable.

        distribution of vs[i], conditioned on vs[j] = val.

        i: index of the variable we want
        j: which variable is conditioned on
        val: the value the jth variable has to have (the conditional value)

        returns: pmf
        """
        # note: returns: pmf of (i | j = val) = the distr of (i) given j = val.
        pmf = PMF(name=name)
        for vs, prob in self.items():
            if vs[j] != val: continue
            pmf.incr(vs[i], prob)

        pmf.normalize()
        return pmf

    def maxLikeInterval(self, percentage=90):
        """Returns the maximum-likelihood credible interval.

        If percentage=90, computes a 90% CI containing the values
        with the highest likelihoods.

        percentage: float between 0 and 100

        Returns: list of values from the suite
        """
        interval = []
        total = 0

        t = [(prob, val) for val, prob in self.items()]
        t.sort(reverse=True)

        for prob, val in t:
            interval.append(val)
            total += prob
            if total >= percentage / 100.0:
                break

        return interval


def makeJoint(pmf1, pmf2):
    """Joint distribution of values from pmf1 and pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        Joint pmf of value pairs
    """
    joint = Joint()
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            joint.set((v1, v2), p1 * p2)
    return joint


def makeHistFromList(t, name=''):
    """Makes a histogram from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this histogram

    Returns:
        Hist object
    """
    hist = Hist(name=name)
    [hist.incr(x) for x in t]
    return hist


def makeHistFromDict(d, name=''):
    """Makes a histogram from a map from values to frequencies.

    Args:
        d: dictionary that maps values to frequencies
        name: string name for this histogram

    Returns:
        Hist object
    """
    return Hist(d, name)


def makePmfFromList(t, name=''):
    """Makes a PMF from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this PMF

    Returns:
        Pmf object
    """
    hist = makeHistFromList(t)  # hist object has stored some random sums in dict d
    d = hist.getDict()
    pmf = PMF(d, name)
    pmf.normalize()
    return pmf


def makePmfFromDict(d, name=''):
    """Makes a PMF from a map from values to probabilities.

    Args:
        d: dictionary that maps values to probabilities
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = PMF(d, name)
    pmf.normalize()
    return pmf


def makePmfFromItems(t, name=''):
    """Makes a PMF from a sequence of value-probability pairs

    Args:
        t: sequence of value-probability pairs
        name: string name for this PMF

    Returns:
        Pmf object
    """
    pmf = PMF(dict(t), name)
    pmf.normalize()
    return pmf


def makePmfFromHist(hist, name=None):
    """Makes a normalized PMF from a Hist object.

    Args:
        hist: Hist object
        name: string name

    Returns:
        Pmf object
    """
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.getDict())
    pmf = PMF(d, name)
    pmf.normalize()
    return pmf


def makePmfFromCdf(cdf, name=None):
    """Makes a normalized Pmf from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Pmf

    Returns:
        Pmf object
    """
    if name is None:
        name = cdf.name

    pmf = PMF(name=name)

    prev = 0.0
    for val, prob in cdf.items():
        pmf.incr(val, prob - prev)
        prev = prob

    return pmf


def makeMixture(metapmf, name='mix'):
    """Make a mixture distribution.

    Args:
      metapmf: Pmf that maps from Pmfs to probs.
      name: string name for the new Pmf.

    Returns: Pmf object.
    """
    mix = PMF(name=name)
    for pmf, p1 in metapmf.items():
        for x, p2 in pmf.items():
            mix.incr(x, p1 * p2)
    return mix


def makeUniformPmf(low, high, n):
    """Make a uniform Pmf.

    low: lowest value (inclusive)
    high: highest value (inclusize)
    n: number of values
    """
    pmf = PMF()
    for x in numpy.linspace(low, high, n):
        pmf.set(x, 1)
    pmf.normalize()
    return pmf


class CDF(object):
    """Represents a cumulative distribution function.

    Attributes:
        xs: sequence of values
        ps: sequence of probabilities
        name: string used as a graph label.
    """

    def __init__(self, xs=None, ps=None, name=''):
        self.xs = [] if xs is None else xs
        self.ps = [] if ps is None else ps
        self.name = name

    def copy(self, name=None):
        """Returns a copy of this Cdf.

        Args:
            name: string name for the new Cdf
        """
        if name is None:
            name = self.name
        return CDF(list(self.xs), list(self.ps), name)

    def makePmf(self, name=None):
        """Makes a Pmf."""
        return makePmfFromCdf(self, name=name)

    def values(self):
        """Returns a sorted list of values.
        """
        return self.xs

    def items(self):
        """Returns a sorted sequence of (value, probability) pairs.

        Note: in Python3, returns an iterator.
        """
        return zip(self.xs, self.ps)

    def append(self, x, p):
        """Add an (x, p) pair to the end of this CDF.

        Note: this us normally used to build a CDF from scratch, not
        to modify existing CDFs.  It is up to the caller to make sure
        that the result is a legal CDF.
        """
        self.xs.append(x)
        self.ps.append(p)

    def shift(self, term):
        """Adds a term to the xs.

        term: how much to add
        """
        new = self.copy()
        new.xs = [x + term for x in self.xs]
        return new

    def scale(self, factor):
        """Multiplies the xs by a factor.

        factor: what to multiply by
        """
        new = self.copy()
        new.xs = [x * factor for x in self.xs]
        return new

    def prob(self, x):
        """Returns CDF(x), the probability that corresponds to value x.

        Args:
            x: number

        Returns:
            float probability
        """
        if x < self.xs[0]: return 0.0
        index = bisect.bisect(self.xs, x)
        p = self.ps[index - 1]
        return p

    def value(self, p):
        """Returns InverseCDF(p), the value that corresponds to probability p.

        Args:
            p: number in the range [0, 1]

        Returns:
            number value
        """
        if p < 0 or p > 1:
            raise ValueError('Probability p must be in range [0, 1]')

        if p == 0: return self.xs[0]
        if p == 1: return self.xs[-1]
        index = bisect.bisect(self.ps, p)
        if p == self.ps[index - 1]:
            return self.xs[index - 1]
        else:
            return self.xs[index]

    def percentile(self, p):
        """Returns the value that corresponds to percentile p.

        Args:
            p: number in the range [0, 100]

        Returns:
            number value
        """
        return self.value(p / 100.0)

    def random(self):
        """Chooses a random value from this distribution."""
        return self.value(random.random())

    def sample(self, n):
        """Generates a random sample from this distribution.
        
        Args:
            n: int length of the sample
        """
        return [self.random() for i in range(n)]

    def mean(self):
        """Computes the mean of a CDF.

        Returns:
            float mean
        """
        old_p = 0
        total = 0.0
        for x, new_p in zip(self.xs, self.ps):
            p = new_p - old_p
            total += p * x
            old_p = new_p
        return total

    def credibleInterval(self, percentage=90):
        """Computes the central credible interval.

        If percentage=90, computes the 90% CI.

        Args:
            percentage: float between 0 and 100

        Returns:
            sequence of two floats, low and high
        """
        prob = (1 - percentage / 100.0) / 2
        interval = self.value(prob), self.value(1 - prob)
        return interval

    def _round(self, multiplier=1000.0):
        """
        An entry is added to the cdf only if the percentile differs
        from the previous value in a significant digit, where the number
        of significant digits is determined by multiplier.  The
        default is 1000, which keeps log10(1000) = 3 significant digits.
        """
        # TODO(write this method)
        raise UnimplementedMethodException()

    def render(self):
        """Generates a sequence of points suitable for plotting.

        An empirical CDF is a step function; linear interpolation
        can be misleading.

        Returns:
            tuple of (xs, ps)
        """
        xs = [self.xs[0]]
        ps = [0.0]
        for i, p in enumerate(self.ps):
            xs.append(self.xs[i])
            ps.append(p)

            try:
                xs.append(self.xs[i + 1])
                ps.append(p)
            except IndexError:
                pass
        return xs, ps

    def max(self, k):  # @ class Cdf
        """Computes the CDF of the maximum of k selections from this dist.

        k: int

        returns: new Cdf
        """
        cdf = self.copy()
        cdf.ps = [p ** k for p in cdf.ps]
        return cdf


def makeCdfFromItems(items, name=''):
    """Makes a cdf from an unsorted sequence of (value, frequency) pairs.

    Args:
        items: unsorted sequence of (value, frequency) pairs
        name: string name for this CDF

    Returns:
        cdf: list of (value, fraction) pairs
    """
    runsum = 0
    xs = []
    cs = []

    for value, count in sorted(items):
        runsum += count
        xs.append(value)
        cs.append(runsum)

    total = float(runsum)
    ps = [c / total for c in cs] #note: dividing again just to be sure it's normalized (a prob dist)

    cdf = CDF(xs, ps, name)
    return cdf


def makeCdfFromDict(d, name=''):
    """Makes a CDF from a dictionary that maps values to frequencies.

    Args:
       d: dictionary that maps values to frequencies.
       name: string name for the data.

    Returns:
        Cdf object
    """
    return makeCdfFromItems(d.iteritems(), name)


def makeCdfFromHist(hist, name=''):
    """Makes a CDF from a Hist object.

    Args:
       hist: Pmf.Hist object
       name: string name for the data.

    Returns:
        Cdf object
    """
    return makeCdfFromItems(hist.items(), name)


def makeCdfFromPmf(pmf, name=None):
    """Makes a CDF from a Pmf object.

    Args:
       pmf: Pmf.Pmf object
       name: string name for the data.

    Returns:
        Cdf object
    """
    if name == None:
        name = pmf.name
    return makeCdfFromItems(pmf.items(), name)


def makeCdfFromList(seq, name=''):
    """Creates a CDF from an unsorted sequence.

    Args:
        seq: unsorted sequence of sortable values
        name: string name for the cdf

    Returns:
       Cdf object
    """
    hist = makeHistFromList(seq)
    return makeCdfFromHist(hist, name)


class UnimplementedMethodException(Exception):
    """Exception if someone calls a method that should be overridden."""


# NOte: this means Suite extends Pmf class
class Suite(PMF):
    """Represents a suite of hypotheses and their probabilities."""

    def update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data

        returns: the normalizing constant
        """
        for hypo in self.keys():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        return self.normalize() # NOTE; different than updateSet() because it normalizes after each updating.

    def logUpdate(self, data):
        """Updates a suite of hypotheses based on new data.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        Note: unlike Update, LogUpdate does not normalize.

        Args:
            data: any representation of the data
        """
        for hypo in self.keys():
            like = self.logLikelihood(data, hypo)
            self.incr(hypo, like)

    def updateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        This is more efficient than calling Update repeatedly because
        it waits until the end to Normalize.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: the normalizing constant
        """
        for data in dataset:
            for hypo in self.keys():
                like = self.likelihood(data, hypo)
                self.mult(hypo, like) # multiply prior by likelihood
        return self.normalize() # NOTE: updates just once after updating with all data.

    def logUpdateSet(self, dataset):
        """Updates each hypothesis based on the dataset.

        Modifies the suite directly; if you want to keep the original, make
        a copy.

        dataset: a sequence of data

        returns: None
        """
        for data in dataset:
            self.logUpdate(data)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()

    def logLikelihood(self, data, hypo):
        """Computes the log likelihood of the data under the hypothesis.

        hypo: some representation of the hypothesis
        data: some representation of the data
        """
        raise UnimplementedMethodException()


    def print(self):
        """Prints the hypotheses and their probabilities."""
        for hypo, prob in sorted(self.items()):
            print("p(", hypo, ") = ", prob, sep="")

    """
    def Print(self, color):  # @ argument 'color' added by me
        #Prints the hypotheses and their probabilities.
        print("For candies with color(s)", color)  # @ added by me
        for hypo, prob in sorted(self.Items()):
            print(hypo, prob)"""


    def makeOdds(self):
        """Transforms from probabilities to odds.

        Values with prob=0 are removed.
        """
        for hypo, prob in self.items():
            if prob:
                self.set(hypo, odds(prob))
            else:
                self.remove(hypo)

    def makeProbs(self):
        """Transforms from odds to probabilities."""
        for hypo, odds in self.items():
            self.set(hypo, probability(odds))


def makeSuiteFromList(t, name=''):
    """Makes a suite from an unsorted sequence of values.

    Args:
        t: sequence of numbers
        name: string name for this suite

    Returns:
        Suite object
    """
    hist = makeHistFromList(t)
    d = hist.getDict()
    return makeSuiteFromDict(d)


def makeSuiteFromHist(hist, name=None):
    """Makes a normalized suite from a Hist object.

    Args:
        hist: Hist object
        name: string name

    Returns:
        Suite object
    """
    if name is None:
        name = hist.name

    # make a copy of the dictionary
    d = dict(hist.getDict())
    return makeSuiteFromDict(d, name)


def makeSuiteFromDict(d, name=''):
    """Makes a suite from a map from values to probabilities.

    Args:
        d: dictionary that maps values to probabilities
        name: string name for this suite

    Returns:
        Suite object
    """
    suite = Suite(name=name)
    suite.setDict(d)
    suite.normalize()
    return suite


def makeSuiteFromCdf(cdf, name=None):
    """Makes a normalized Suite from a Cdf object.

    Args:
        cdf: Cdf object
        name: string name for the new Suite

    Returns:
        Suite object
    """
    if name is None:
        name = cdf.name

    suite = Suite(name=name)

    prev = 0.0
    for val, prob in cdf.items():
        suite.incr(val, prob - prev)
        prev = prob

    return suite


class PDF(object):
    """Represents a probability density function (PDF)."""

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        raise UnimplementedMethodException()

    def makePmf(self, xs, name=''):
        """Makes a discrete version of this Pdf, evaluated at xs.

        xs: equally-spaced sequence of values

        Returns: new Pmf
        """
        pmf = PMF(name=name)
        for x in xs:
            pmf.set(x, self.density(x))
        pmf.normalize()
        return pmf


class GaussianPDF(PDF):
    """Represents the PDF of a Gaussian distribution."""

    def __init__(self, mu, sigma):
        """Constructs a Gaussian Pdf with given mu and sigma.

        mu: mean
        sigma: standard deviation
        """
        self.mu = mu
        self.sigma = sigma

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        return evalGaussianPdf(x, self.mu, self.sigma)


class EstimatedPDF(PDF):
    """Represents a PDF estimated by KDE."""

    def __init__(self, sample):
        """Estimates the density function based on a sample.

        sample: sequence of data
        """
        self.kde = scipy.stats.gaussian_kde(sample)

    def density(self, x):
        """Evaluates this Pdf at x.

        Returns: float probability density
        """
        return self.kde.evaluate(x)

    def makePmf(self, xs, name=''):
        ps = self.kde.evaluate(xs)
        pmf = makePmfFromItems(zip(xs, ps), name=name)
        return pmf


def percentile(pmf, percentage):
    """Computes a percentile of a given Pmf.

    percentage: float 0-100
    """
    p = percentage / 100.0
    total = 0
    for val, prob in pmf.items():
        total += prob
        if total >= p:
            return val # NOTE: find the value key that corresponds to percentile probability.


def credibleInterval(pmf, percentage=90):
    """Computes a credible interval for a given distribution.

    If percentage=90, computes the 90% CI.

    Args:
        pmf: Pmf object representing a posterior distribution
        percentage: float between 0 and 100

    Returns:
        sequence of two floats, low and high
    """
    cdf = pmf.makeCdf()
    prob = (1 - percentage / 100.0) / 2
    interval = cdf.value(prob), cdf.value(1 - prob)
    return interval


def pmfProbLess(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 < v2:
                total += p1 * p2
    return total


def pmfProbGreater(pmf1, pmf2):
    """Probability that a value from pmf1 is less than a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 > v2:
                total += p1 * p2
    return total


def pmfProbEqual(pmf1, pmf2):
    """Probability that a value from pmf1 equals a value from pmf2.

    Args:
        pmf1: Pmf object
        pmf2: Pmf object

    Returns:
        float probability
    """
    total = 0.0
    for v1, p1 in pmf1.items():
        for v2, p2 in pmf2.items():
            if v1 == v2:
                total += p1 * p2
    return total


def randomSum(dists):
    """Chooses a random value from each dist and returns the sum.

    dists: sequence of Pmf or Cdf objects

    returns: numerical sum
    """
    total = sum(dist.random() for dist in dists)
    return total


def sampleSum(dists, n):
    """Draws a sample of sums from a list of distributions.

    dists: sequence of Pmf or Cdf objects
    n: sample size

    returns: new Pmf of sums
    """
    pmf = makePmfFromList(randomSum(dists) for i in range(n))
    return pmf


def evalGaussianPdf(x, mu, sigma):
    """Computes the unnormalized PDF of the normal distribution.

    x: value
    mu: mean
    sigma: standard deviation
    
    returns: float probability density
    """
    return scipy.stats.norm.pdf(x, mu, sigma)


def makeGaussianPmf(mu, sigma, num_sigmas, n=201):
    """Makes a PMF discrete approx to a Gaussian distribution.
    
    mu: float mean
    sigma: float standard deviation
    num_sigmas: how many sigmas to extend in each direction
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    low = mu - num_sigmas * sigma
    high = mu + num_sigmas * sigma

    for x in numpy.linspace(low, high, n):
        p = evalGaussianPdf(x, mu, sigma)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def evalBinomialPmf(k, n, p):
    """Evaluates the binomial pmf.

    Returns the probabily of k successes in n trials with probability p.
    """
    return scipy.stats.binom.pmf(k, n, p)
    

def evalPoissonPmf(k, lmbda):
    """Computes the Poisson PMF.

    k: number of events
    lam: parameter lambda in events per unit time

    returns: float probability
    """
    # don't use the scipy function (yet).  for lam=0 it returns NaN;
    # should be 0.0
    # return scipy.stats.poisson.pmf(k, lam)

    return lmbda ** k * math.exp(-lmbda) / math.factorial(k)


def makePoissonPmf(lam, high, step=1):
    """Makes a PMF discrete approx to a Poisson distribution.

    lam: parameter lambda in events per unit time
    high: upper bound of the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    for k in range(0, high + 1, step):
        p = evalPoissonPmf(k, lam)
        pmf.set(k, p)
    pmf.normalize()
    return pmf


def evalExponentialPdf(x, lam):
    """Computes the exponential PDF.

    x: value
    lam: parameter lambda in events per unit time

    returns: float probability density
    """
    return lam * math.exp(-lam * x)


def evalExponentialCdf(x, lam):
    """Evaluates CDF of the exponential distribution with parameter lam."""
    return 1 - math.exp(-lam * x)


def makeExponentialPmf(lam, high, n=200):
    """Makes a PMF discrete approx to an exponential distribution.

    lam: parameter lambda in events per unit time
    high: upper bound
    n: number of values in the Pmf

    returns: normalized Pmf
    """
    pmf = PMF()
    for x in numpy.linspace(0, high, n):
        p = evalExponentialPdf(x, lam)
        pmf.set(x, p)
    pmf.normalize()
    return pmf


def standardGaussianCdf(x):
    """Evaluates the CDF of the standard Gaussian distribution.
    
    See http://en.wikipedia.org/wiki/Normal_distribution
    #Cumulative_distribution_function

    Args:
        x: float
                
    Returns:
        float
    """
    return (erf(x / ROOT2) + 1) / 2


def gaussianCdf(x, mu=0, sigma=1):
    """Evaluates the CDF of the gaussian distribution.
    
    Args:
        x: float

        mu: mean parameter
        
        sigma: standard deviation parameter
                
    Returns:
        float
    """
    return standardGaussianCdf(float(x - mu) / sigma)


def gaussianCdfInverse(p, mu=0, sigma=1):
    """Evaluates the inverse CDF of the gaussian distribution.

    See http://en.wikipedia.org/wiki/Normal_distribution#Quantile_function  

    Args:
        p: float

        mu: mean parameter
        
        sigma: standard deviation parameter
                
    Returns:
        float
    """
    x = ROOT2 * erfinv(2 * p - 1)
    return mu + x * sigma


class Beta(object):
    """Represents a Beta distribution.

    See http://en.wikipedia.org/wiki/Beta_distribution
    """
    def __init__(self, alpha=1, beta=1, name=''):
        """Initializes a Beta distribution."""
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def update(self, data):
        """Updates a Beta distribution.

        data: pair of int (heads, tails)
        """
        heads, tails = data
        self.alpha += heads
        self.beta += tails

    def mean(self):
        """Computes the mean of this distribution."""
        return float(self.alpha) / (self.alpha + self.beta)

    def random(self):
        """Generates a random variate from this distribution."""
        return random.betavariate(self.alpha, self.beta)

    def sample(self, n):
        """Generates a random sample from this distribution.

        n: int sample size
        """
        size = n,
        return numpy.random.beta(self.alpha, self.beta, size)

    def evalPdf(self, x):
        """Evaluates the PDF at x."""
        return x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)

    def makePmf(self, steps=101, name=''):
        """Returns a Pmf of this distribution.

        Note: Normally, we just evaluate the PDF at a sequence
        of points and treat the probability density as a probability
        mass.

        But if alpha or beta is less than one, we have to be
        more careful because the PDF goes to infinity at x=0
        and x=1.  In that case we evaluate the CDF and compute
        differences.
        """
        if self.alpha < 1 or self.beta < 1:
            cdf = self.makeCdf()
            pmf = cdf.makePmf()
            return pmf

        xs = [i / (steps - 1.0) for i in range(steps)]
        probs = [self.evalPdf(x) for x in xs]
        pmf = makePmfFromDict(dict(zip(xs, probs)), name)
        return pmf

    def makeCdf(self, steps=101):
        """Returns the CDF of this distribution."""
        xs = [i / (steps - 1.0) for i in range(steps)]
        ps = [scipy.special.betainc(self.alpha, self.beta, x) for x in xs]
        cdf = CDF(xs, ps)
        return cdf


class Dirichlet(object):
    """Represents a Dirichlet distribution.

    See http://en.wikipedia.org/wiki/Dirichlet_distribution
    """

    def __init__(self, n, conc=1, name=''):
        """Initializes a Dirichlet distribution.

        n: number of dimensions
        conc: concentration parameter (smaller yields more concentration)
        name: string name
        """
        if n < 2:
            raise ValueError('A Dirichlet distribution with '
                             'n<2 makes no sense')

        self.n = n
        self.params = numpy.ones(n, dtype=numpy.float) * conc
        self.name = name

    def update(self, data):
        """Updates a Dirichlet distribution.

        data: sequence of observations, in order corresponding to params
        """
        m = len(data)
        self.params[:m] += data

    def random(self):
        """Generates a random variate from this distribution.

        Returns: normalized vector of fractions
        """
        p = numpy.random.gamma(self.params)
        return p / p.sum()

    def likelihood(self, data):
        """Computes the likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float probability
        """
        m = len(data)
        if self.n < m:
            return 0

        x = data
        p = self.random()
        q = p[:m] ** x
        return q.prod()

    def logLikelihood(self, data):
        """Computes the log likelihood of the data.

        Selects a random vector of probabilities from this distribution.

        Returns: float log probability
        """
        m = len(data)
        if self.n < m:
            return float('-inf')

        x = self.random()
        y = numpy.log(x[:m]) * data
        return y.sum()

    def marginalBeta(self, i):
        """Computes the marginal distribution of the ith element.

        See http://en.wikipedia.org/wiki/Dirichlet_distribution
        #Marginal_distributions

        i: int

        Returns: Beta object
        """
        alpha0 = self.params.sum()
        alpha = self.params[i]
        return Beta(alpha, alpha0 - alpha)

    def predictivePmf(self, xs, name=''):
        """Makes a predictive distribution.

        xs: values to go into the Pmf

        Returns: Pmf that maps from x to the mean prevalence of x
        """
        alpha0 = self.params.sum()
        ps = self.params / alpha0
        return makePmfFromItems(zip(xs, ps), name=name)


def binomialCoef(n, k):
    """Compute the binomial coefficient "n choose k".

    n: number of trials
    k: number of successes

    Returns: float
    """
    return scipy.misc.comb(n, k)


def logBinomialCoef(n, k):
    """Computes the log of the binomial coefficient.

    http://math.stackexchange.com/questions/64716/
    approximating-the-logarithm-of-the-binomial-coefficient

    n: number of trials
    k: number of successes

    Returns: float
    """
    return n * math.log(n) - k * math.log(k) - (n - k) * math.log(n - k)










# -----------------------TODO: from thinkbayes 2
'''
class FixedWidthVariables(object):
    """Represents a set of variables in a fixed width file."""

    def __init__(self, variables, index_base=0):
        """Initializes.

        variables: DataFrame
        index_base: are the indices 0 or 1 based?

        Attributes:
        colspecs: list of (start, end) index tuples
        names: list of string variable names
        """
        self.variables = variables

        # note: by default, subtract 1 from colspecs
        self.colspecs = variables[['start', 'end']] - index_base

        # convert colspecs to a list of pair of int
        self.colspecs = self.colspecs.astype(numpy.int).values.tolist()
        self.names = variables['name']

    def ReadFixedWidth(self, filename, **options):
        """Reads a fixed width ASCII file.

        filename: string filename

        returns: DataFrame
        """
        df = pandas.read_fwf(filename,
                             colspecs=self.colspecs,
                             names=self.names,
                             **options)
        return df




def TrimmedMeanVar(t, p=0.01):
    """Computes the trimmed mean and variance of a sequence of numbers.

    Side effect: sorts the list.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        float
    """
    t = Trim(t, p)
    mu, var = MeanVar(t)
    return mu, var


def MeanVar(xs, ddof=0):
    """Computes mean and variance.

    Based on http://stackoverflow.com/questions/19391149/
    numpy-mean-and-variance-from-single-function

    xs: sequence of values
    ddof: delta degrees of freedom

    returns: pair of float, mean and var
    """
    xs = numpy.asarray(xs)
    mean = xs.mean()
    s2 = Var(xs, mean, ddof)
    return mean, s2


def Trim(t, p=0.01):
    """Trims the largest and smallest elements of t.

    Args:
        t: sequence of numbers
        p: fraction of values to trim off each end

    Returns:
        sequence of values
    """
    n = int(p * len(t))
    t = sorted(t)[n:-n]
    return t


def Var(xs, mu=None, ddof=0):
    """Computes variance.

    xs: sequence of values
    mu: option known mean
    ddof: delta degrees of freedom

    returns: float
    """
    xs = numpy.asarray(xs)

    if mu is None:
        mu = xs.mean()

    ds = xs - mu
    return numpy.dot(ds, ds) / (len(xs) - ddof)

'''