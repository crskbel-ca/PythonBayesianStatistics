"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from src.thinkbayes import PMF


class Cookie(PMF): # Cookie extends Pmf
    """A map from string bowl ID to probability."""

    def __init__(self, hypos):
        """Initialize self.

        hypos: sequence of string bowl IDs
        """
        PMF.__init__(self)
        for hypo in hypos:
            self.set(hypo, 1)
        self.normalize()

    def update(self, data):
        """Updates the PMF with new data.

        data: string cookie type
        """
        for hypo in self.keys():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        self.normalize()


    mixes = {
        'Bowl 1':dict(vanilla=0.75, chocolate=0.25),
        'Bowl 2':dict(vanilla=0.5, chocolate=0.5),
        }

    def likelihood(self, data, hypo):
        """The likelihood of the data under the hypothesis.

        data: string cookie type
        hypo: string bowl ID
        """
        mix = self.mixes[hypo]
        like = mix[data]
        return like


def main():
    hypos = ['Bowl 1', 'Bowl 2']

    pmf = Cookie(hypos)

    # Generalizing when drawing more than one cookie in these orders from the bowls (with replacement)
    # todo is this binomial because no replacement? Where does it fit in?
    dataset = ['vanilla', 'chocolate', 'vanilla']
    for data in dataset:
        pmf.update(data)

    for hypo, prob in pmf.items():
        print(hypo, ": ", prob, sep="")


if __name__ == '__main__':
    main()
