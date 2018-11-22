"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from src.thinkbayes import Suite


class Dice(Suite):
    """Represents hypotheses about which die was rolled."""

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: integer number of sides on the die
        data: integer die roll
        """
        if hypo < data:
            return 0
        else:
            return 1.0/hypo


def main():
    ### QUESTION TO SOLVE: select a die from box at random, roll it and get a 6. What is the probability
    ### that each die was rolled? (prob of having rolled 4-sided, 6-sided, 8, 12, 20-sided dice).

    # HYPOTHESIS: 4,6,8,12,20-sided dice in tbe box
    # DATA: numbers 1 - 20 for respective dice.
    # LIKELIHOOD: if hypo<data then roll is greater than numsides on the die => likelihood = 0.
    # But otherwise, given hypo sides, chance of rolling data (element, one side) is 1/hypo, regardless of data.

    # Step 1: Prior: The 4-sided, 6-sided, 8, 12, 20 sided dice in the box.
    suite = Dice([4, 6, 8, 12, 20]) # step 1: setting all these hypos to have equal probability: 1/5

    suite.update(6) # Step 2: updating the numerator with likelihood, then normalizing.
    print('After one 6')
    suite.printSuite()

    # Note: the p(4) and p(6) become 0 because they are less than some data here.
    for roll in [6, 8, 7, 7, 5, 4]:
        suite.update(roll)

    # NOTE: the 8 has highest probability because it has lowest num sides after 4 and 6 which got eliminated,
    # and likewise, 20 has smallest chance because it has most sides.
    print('After more rolls')
    suite.printSuite()


if __name__ == '__main__':
    main()
