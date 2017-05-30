"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from thinkbayes import Suite



# Program first finds probability for Y94 and Y96 ==> A = 0.588, B = 0.411,
#  then for (Y94, G96) ==> 0.740 and (Y96, G94) ==> 0.259

class MandM(Suite):
    """Map from hypothesis (A or B) to probability."""

    mix94 = dict(brown=30,
                 yellow=20,
                 red=20,
                 green=10,
                 orange=10,
                 tan=10,
                 blue=0)

    mix96 = dict(blue=24,
                 green=20,
                 orange=16,
                 yellow=14,
                 red=13,
                 brown=13,
                 tan=0)

    hypoA = dict(bag1=mix94, bag2=mix96)
    hypoB = dict(bag1=mix96, bag2=mix94)

    hypotheses = dict(A=hypoA, B=hypoB)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: string hypothesis (A or B)
        data: tuple of string bag, string color
        """
        bag, color = data
        mix = self.hypotheses[hypo][bag]
        like = mix[color]
        return like


def main():
    suite = MandM('AB')

    # note: saying update bag1/yellow and then bag2/green is equivalent to finding
    # P(A | Y & G) and P(B | Y & G)
    suite.update(('bag1', 'yellow')) # data = bag1, yellow
    suite.print()
    suite.update(('bag2', 'green'))
    suite.print()


if __name__ == '__main__':
    main()
