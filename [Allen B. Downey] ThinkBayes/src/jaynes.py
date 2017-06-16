"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com
Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import thinkbayes

from src import thinkplot

"""This file contains a solution to an exercise from Think Bayes,
by Allen B. Downey
I got the idea from Tom Campbell-Ricketts author of the Maximum
Entropy blog at
http://maximum-entropy-blog.blogspot.com
And he got the idea from E.T. Jaynes, author of the classic
_Probability Theory: The Logic of Science_.
Here's the version from Think Bayes:
Radioactive decay is well modeled by a Poisson process; the
probability that an atom decays is the same at any point in time.
Suppose that a radioactive source emits particles toward a Geiger
counter at an average rate of $r$ particles per second, but the
counter only registers a fraction, $f$, of the particles that hit it.
If $f$ is 10\% and the counter registers 15 particles in a one second
interval, what is the posterior distribution of $n$, the actual number
of particles that hit the counter, and $p$, the average rate particles
are emitted?
"""

FORMATS = ['pdf'] # , 'eps', 'png'

class Emitter(thinkbayes.Suite): # note: meta suite: contains detectors mapped to probabilities (0.01 initially)
    """Represents hypotheses about r."""

    def __init__(self, rs, f=0.1):
        """Initializes the Suite.
        rs: sequence of hypothetical emission rates
        f: fraction of particles registered
        """
        detectors = [Detector(r, f) for r in rs]
        thinkbayes.Suite.__init__(self, detectors) # and normalizes all 100 of the detectors (0.01)

    def update(self, data):
        """Updates the Suite based on data.
        data: number of particles counted
        """
        thinkbayes.Suite.update(self, data) # note: updating meta pmf

        for detector in self.keys(): # note changed from values to keys, since detectors are keys.
            detector.update(data) # note: updating each detector pmf in turn.

    def likelihood(self, data, hypo):
        """Likelihood of the data given the hypothesis.
        Args:
            data: number of particles counted
            hypo: emission rate, r
        Returns:
            probability density of the data under the hypothesis
        """
        detector = hypo
        like = detector.suiteLikelihood(data)
        return like

    def distOfR(self, name=''):
        """Returns the PMF of r."""
        items = [(detector.r, prob) for detector, prob in self.items()]
        return thinkbayes.makePmfFromItems(items, name=name)

    def distOfN(self, name=''):
        """Returns the PMF of n."""
        return thinkbayes.makeMixture(self, name=name)


class Emitter2(thinkbayes.Suite):
    """Represents hypotheses about r."""

    def __init__(self, rs, f=0.1):
        """Initializes the Suite.
        rs: sequence of hypothetical emission rates
        f: fraction of particles registered
        """
        detectors = [Detector(r, f) for r in rs]
        thinkbayes.Suite.__init__(self, detectors)

    def likelihood(self, data, hypo):
        """Likelihood of the data given the hypothesis.
        Args:
            data: number of counted per unit time
            hypo: emission rate, r
        Returns:
            probability density of the data under the hypothesis
        """
        return hypo.update(data)
        # note: hypo=detector pmf (goes into update of suite, so same thing achieved as Emitter's update)

    def distOfR(self, name=''):
        """Returns the PMF of r."""
        items = [(detector.r, prob) for detector, prob in self.items()]
        return thinkbayes.makePmfFromItems(items, name=name)

    def distOfN(self, name=''):
        """Returns the PMF of n."""
        return thinkbayes.makeMixture(self, name=name)


class Detector(thinkbayes.Suite): # note: Detector is suite is pmf
    """Represents hypotheses about n."""

    def __init__(self, r, f, high=500, step=5):
        """Initializes the suite.
        r: known emission rate, r
        f: fraction of particles registered
        high: maximum number of particles, n
        step: step size between hypothetical values of n
        """
        pmf = thinkbayes.makePoissonPmf(r, high, step=step)
        thinkbayes.Suite.__init__(self, pmf, name=r)
        self.r = r # the emission rate, (changes)
        self.f = f # fraction of particles hit (always 0.1)

    def likelihood(self, data, hypo):
        """Likelihood of the data given the hypothesis.
        data: number of particles counted
        hypo: number of particles hitting the counter, n
        """
        k = data
        n = hypo
        p = self.f
        # note: given n particles, p=0.1 single particle hit, what is prob of hitting k=15 particles?
        return thinkbayes.evalBinomialPmf(k, n, p)

    def suiteLikelihood(self, data):
        """Adds up the total probability of the data under the suite.
        data: number of particles counted
        """
        total = 0
        for hypo, prob in self.items():
            like = self.likelihood(data, hypo)
            total += prob * like # todo: is this sort of a mixture probability?
        return total


def main():
    # note: METHOD 1: assume that f = 0.1 is known! ------------------------
    k = 15
    f = 0.1

    # plot Detector suites for a range of hypothetical r
    thinkplot.prePlot(num=3)
    for r in [100, 250, 400]:
        suite = Detector(r, f, step=1)
        suite.update(k)
        thinkplot.pmf(suite)
        print(suite.maximumLikelihood())

    thinkplot.save(root='jaynes1',
                   xlabel='Number of particles (n)',
                   ylabel='PMF',
                   formats=FORMATS)


    # note: METHOD 2: k is unknown. ---------------------------------------
    # plot the posterior distributions of r and n
    hypos = range(1, 501, 5)
    suite = Emitter(hypos, f=f)
    suite.update(k)

    thinkplot.prePlot(num=2)
    post_r = suite.distOfR(name='posterior r')
    post_n = suite.distOfN(name='posterior n')

    thinkplot.pmf(post_r)
    thinkplot.pmf(post_n)

    thinkplot.save(root='jaynes2',
                   xlabel='Emission rate',
                   ylabel='PMF',
                   formats=FORMATS)


    # note: BUILDING ON 2: optimization in Emitter's likelihood.
    hypos = range(1, 501, 5)
    suite = Emitter2(hypos, f=f)
    suite.update(k)

    thinkplot.prePlot(num=2)
    post_r = suite.distOfR(name='posterior r')
    post_n = suite.distOfN(name='posterior n')

    thinkplot.pmf(post_r)
    thinkplot.pmf(post_n)

    thinkplot.save(root='jaynes3',
                   xlabel='Emission rate',
                   ylabel='PMF',
                   formats=FORMATS)

if __name__ == '__main__':
    main()