"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from thinkbayes import PMF


class Monty(PMF):
    """Map from string location of car to probability"""

    def __init__(self, hypos):
        """Initialize the distribution.

        hypos: sequence of hypotheses
        """
        PMF.__init__(self) # todo: is this like super in java?
        for hypo in hypos:
            self.set(hypo, 1)
        self.normalize()

    def update(self, data):
        """Updates each hypothesis based on the data.

        data: any representation of the data
        """
        for hypo in self.keys():
            like = self.likelihood(data, hypo)
            self.mult(hypo, like)
        self.normalize()

    def likelihood(self, data, hypo):
        """Compute the likelihood of the data under the hypothesis.

        hypo: string name of the door where the prize is
        data: string name of the door Monty opened
        """
        if hypo == data:  #then hypo = B
            return 0
        elif hypo == 'A':
            return 0.5
        else: # hypo = C
            return 1


def main():
    hypos = 'ABC'
    pmf = Monty(hypos)

    # Stating that door to be chosen is B.
    data = 'B'
    pmf.update(data)

    for hypo, prob in sorted(pmf.items()):
        print(hypo, ": ", prob, sep="")


if __name__ == '__main__':
    main()
