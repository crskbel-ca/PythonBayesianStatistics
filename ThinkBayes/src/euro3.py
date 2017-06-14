"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

"""This file contains a partial solution to a problem from
MacKay, "Information Theory, Inference, and Learning Algorithms."

    Exercise 3.15 (page 50): A statistical statement appeared in
    "The Guardian" on Friday January 4, 2002:

        When spun on edge 250 times, a Belgian one-euro coin came
        up heads 140 times and tails 110.  'It looks very suspicious
        to me,' said Barry Blight, a statistics lecturer at the London
        School of Economics.  'If the coin were unbiased, the chance of
        getting a result as extreme as that would be less than 7%.'

MacKay asks, "But do these data give evidence that the coin is biased
rather than fair?"

"""

from src import thinkbayes


class Euro(thinkbayes.Suite):
    """Represents hypotheses about the probability of heads."""

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer value of x, the probability of heads (0-100)
        data: tuple of (number of heads, number of tails)
        """
        x = hypo / 100.0
        heads, tails = data
        like = x**heads * (1-x)**tails
        return like


def trianglePrior():
    """Makes a Suite with a triangular prior."""
    suite = Euro()
    for x in range(0, 51):
        suite.set(x, x)
    for x in range(51, 101):
        suite.set(x, 100 - x)
    suite.normalize()
    return suite


def suiteLikelihood(suite, data):
    """Computes the weighted average of likelihoods for sub-hypotheses.

    suite: Suite that maps sub-hypotheses to probability
    data: some representation of the data

    returns: float likelihood
    """
    total = 0
    for hypo, prob in suite.items():
        like = suite.likelihood(data, hypo)
        total += prob * like
    return total


def main():
    data = 140, 110
    #data = 8, 12


    # note: APPROACH 1: using data (bogus)
    # note: Calculating probability of data assuming fair coin
    suite = Euro()
    like_f = suite.likelihood(data, 50) # hypo = 50 to make percentage 1/2
    print('p(D|F) =', like_f)
    print("")

    # note: Calculating probability of data assuming biased coin
    actual_percent = 100.0 * 140 / 250
    likelihood = suite.likelihood(data, actual_percent)
    print('p(D|B_cheat)          =', likelihood)
    print('p(D|B_cheat) / p(D|F) =', likelihood / like_f) # Bayes factor > 1 so coin is biased.
    print("")

    # note: APPROACH 2: Do not look at data first. Now we are just not sure which way the coin
    # is biased. If p(heads) = 0.6 or 0.4, so we take both hypos and average their likelihoods.
    like40 = suite.likelihood(data, 40)
    like60 = suite.likelihood(data, 60)
    likelihood = 0.5 * like40 + 0.5 * like60
    print('p(D|B_two)          =', likelihood)
    print('p(D|B_two) / p(D|F) =', likelihood / like_f) # using the either-hypo approach rather than data approach
    print("") # and we see Bayes factor still > 1, though weak evidence that coin is biased.

    # note: APPROACH 3: following from approach 2 but not sure what prob is (0.4 or 0.6 or anything)
    b_uniform = Euro(range(0, 101))
    b_uniform.remove(50) # because that would mean we can assume coin is fair (trying to prove it's not, but removing
    #  makes almost no difference)
    b_uniform.normalize()
    likelihood = suiteLikelihood(b_uniform, data)
    likelihood_same = b_uniform.update(data)
    print('p(D|B_uniform)          = {0} and same {1}'.format(likelihood, likelihood_same))
    print('p(D|B_uniform)          =', likelihood)
    print('p(D|B_uniform) / p(D|F) =', likelihood / like_f) # weak evidence
    print("")

    # Triangle prior gives more weight to values near 50.
    b_tri = trianglePrior()
    b_tri.remove(50)
    b_tri.normalize()
    likelihood = b_tri.update(data)
    print('p(D|B_tri)          =', likelihood)
    print('p(D|B_tri) / p(D|F) =', likelihood / like_f)

    # Depending on which definition we choose, the data might provide evidence for or
    # against the hypothesis that the coin is biased, but in either case it is relatively weak
    # evidence.

if __name__ == '__main__':
    main()
