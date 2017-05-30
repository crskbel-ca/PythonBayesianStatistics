"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2013 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import csv
import numpy
import thinkbayes
import thinkplot

import matplotlib.pyplot as pyplot


FORMATS = ['png', 'pdf', 'eps']


def readData(filename='showcases.2011.csv'):
    """Reads a CSV file of data.

    Args:
      filename: string filename

    Returns: sequence of (price1 price2 bid1 bid2 diff1 diff2) tuples
    """
    fp = open(filename)
    reader = csv.reader(fp)
    res = []

    for t in reader:
        _heading = t[0]
        data = t[1:]
        try:
            data = [int(x) for x in data]
            print(_heading, data[0], len(data))
            res.append(data)
        except ValueError:
            pass

    fp.close()
    return zip(*res)
    





class Price(thinkbayes.Suite):
    """Represents hypotheses about the price of a showcase."""

    def __init__(self, pmf, player, name=''):
        """Constructs the suite.

        pmf: prior distribution of price
        player: Player object
        name: string
        """
        thinkbayes.Suite.__init__(self, pmf, name=name)
        self.player = player

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        hypo: actual price
        data: the contestant's guess
        """
        price = hypo
        guess = data

        error = price - guess
        like = self.player.errorDensity(error)

        return like


class GainCalculator(object):
    """Encapsulates computation of expected gain."""

    def __init__(self, player, opponent):
        """Constructs the calculator.

        player: Player
        opponent: Player
        """
        self.player = player
        self.opponent = opponent

    def expectedGains(self, low=0, high=75000, n=101):
        """Computes expected gains for a range of bids.

        low: low bid
        high: high bid
        n: number of bids to evaluates

        returns: tuple (sequence of bids, sequence of gains)
    
        """
        bids = numpy.linspace(low, high, n)

        gains = [self.expectedGain(bid) for bid in bids]

        return bids, gains

    def expectedGain(self, bid):
        """Computes the expected return of a given bid.

        bid: your bid
        """
        suite = self.player.posterior
        total = 0
        for price, prob in sorted(suite.items()):
            gain = self.gain(bid, price)
            total += prob * gain
        return total

    def gain(self, bid, price):
        """Computes the return of a bid, given the actual price.

        bid: number
        price: actual price
        """
        # if you overbid, you get nothing
        if bid > price:
            return 0

        # otherwise compute the probability of winning
        diff = price - bid
        prob = self.probWin(diff)

        # if you are within 250 dollars, you win both showcases
        if diff <= 250:
            return 2 * price * prob
        else:
            return price * prob

    def probWin(self, diff):
        """Computes the probability of winning for a given diff.

        diff: how much your bid was off by
        """
        prob = (self.opponent.probOverbid() +
                self.opponent.probWorseThan(diff))
        return prob




class Player(object):
    """Represents a player on The Price is Right."""

    n = 101
    price_xs = numpy.linspace(0, 75000, n)

    def __init__(self, prices, bids, diffs):
        """Construct the Player.

        prices: sequence of prices
        bids: sequence of bids
        diffs: sequence of underness (negative means over)
        """
        self.pdf_price = thinkbayes.EstimatedPDF(prices)
        self.cdf_diff = thinkbayes.makeCdfFromList(diffs)

        mu = 0
        sigma = numpy.std(diffs)
        self.pdf_error = thinkbayes.GaussianPDF(mu, sigma)

    def errorDensity(self, error):
        """Density of the given error in the distribution of error.

        error: how much the bid is under the actual price
        """
        return self.pdf_error.density(error)

    def pmfPrice(self):
        """Returns a new Pmf of prices.

        A discrete version of the estimated Pdf.
        """
        return self.pdf_price.makePmf(self.price_xs)

    def cdfDiff(self):
        """Returns a reference to the Cdf of differences (underness).
        """
        return self.cdf_diff

    def probOverbid(self):
        """Returns the probability this player overbids.
        """
        return self.cdf_diff.prob(-1) # overbid means negative xs so take prob percentile at rightmost neg x

    def probWorseThan(self, diff):
        """Probability this player's diff is greater than the given diff.

        diff: how much the oppenent is off by (always positive)
        """
        return 1 - self.cdf_diff.prob(diff) # probability that opponent is off by more than diff
        # so return 1 - prob(diff)

    def makeBeliefs(self, guess):
        """Makes a posterior distribution based on estimated price.

        Sets attributes prior and posterior.

        guess: what the player thinks the showcase is worth        
        """
        pmf = self.pmfPrice() # return discrete version of estimated PDF of prices
        self.prior = Price(pmf, self, name='prior')
        self.posterior = self.prior.copy(name='posterior')
        self.posterior.update(guess)

    def optimalBid(self, guess, opponent):
        """Computes the bid that maximizes expected return.
        
        guess: what the player thinks the showcase is worth 
        opponent: Player

        Returns: (optimal bid, expected gain)
        """
        self.makeBeliefs(guess)
        calc = GainCalculator(self, opponent)
        bids, gains = calc.expectedGains()
        gain, bid = max(zip(gains, bids))
        return bid, gain

    def plotBeliefs(self, root):
        """Plots prior and posterior beliefs.

        root: string filename root for saved figure
        """
        thinkplot.clf()
        thinkplot.prePlot(num=2)
        thinkplot.pmfs([self.prior, self.posterior])
        thinkplot.save(root=root,
                       xlabel='price ($)',
                       ylabel='PMF',
                       formats=FORMATS)


def makePlots(player1, player2):
    """Generates two plots.

    price1 shows the priors for the two players
    price2 shows the distribution of diff for the two players
    """

    # plot the prior distribution of price for both players
    thinkplot.clf()
    thinkplot.prePlot(num=2)
    pmf1 = player1.pmfPrice()
    pmf1.name = 'showcase 1'
    pmf2 = player2.pmfPrice()
    pmf2.name = 'showcase 2'
    thinkplot.pmfs([pmf1, pmf2])
    thinkplot.save(root='price1_showcase1,2_priorPmfs',
                   xlabel='price ($)',
                   ylabel='PDF',
                   formats=FORMATS)

    # plot the historical distribution of underness for both players
    thinkplot.clf()
    thinkplot.prePlot(num=2)
    cdf1 = player1.cdfDiff()
    cdf1.name = 'player 1'
    cdf2 = player2.cdfDiff()
    cdf2.name = 'player 2'

    print('\n\nPlayer median', cdf1.percentile(50))
    print('Player median', cdf2.percentile(50))

    print('\nPlayer 1 overbids', player1.probOverbid())
    print('Player 2 overbids', player2.probOverbid())

    thinkplot.cdfs([cdf1, cdf2])
    thinkplot.save(root='price2_diffs_cdf',
                   xlabel='diff ($)',
                   ylabel='CDF',
                   formats=FORMATS)


def makePlayers():
    """Reads data and makes player objects."""
    print("READING ... 2011 showcases: ")
    data1 = readData(filename='showcases.2011.csv')
    print("\nREADING ... 2012 showcases: ")
    data2 = readData(filename='showcases.2012.csv')
    data = list(data1) + list(data2) #@fix used lists as a workaround for adding zip objects
    # http://stackoverflow.com/questions/1071201/why-does-list-comprehension-using-a-zip-object-results-in-an-empty-list

    cols = zip(*data)
    price1, price2, bid1, bid2, diff1, diff2 = cols

    # print list(sorted(price1))
    # print len(price1)

    player1 = Player(price1, bid1, diff1)
    player2 = Player(price2, bid2, diff2)

    return player1, player2


def plotExpectedGains(guess1=20000, guess2=40000):
    """Plots expected gains as a function of bid.

    guess1: player1's estimate of the price of showcase 1
    guess2: player2's estimate of the price of showcase 2
    """
    player1, player2 = makePlayers()
    makePlots(player1, player2)

    player1.makeBeliefs(guess1)
    player2.makeBeliefs(guess2)

    print('\n\nPlayer 1 prior mle', player1.prior.maximumLikelihood())
    print('Player 2 prior mle', player2.prior.maximumLikelihood())
    print('\nPlayer 1 mean', player1.posterior.mean())
    print('Player 2 mean', player2.posterior.mean())
    print('\nPlayer 1 mle', player1.posterior.maximumLikelihood())
    print('Player 2 mle', player2.posterior.maximumLikelihood())

    player1.plotBeliefs('price3_prior,posterior_player1') # was price3
    player2.plotBeliefs('price4_prior,posterior_player2') # was price4

    calc1 = GainCalculator(player1, player2)
    calc2 = GainCalculator(player2, player1)

    thinkplot.clf()
    thinkplot.prePlot(num=2)

    # NOTE: player 1 optimal bid = 21,000, expgain =  16,700, best guesss = 20,000
    bids, gains = calc1.expectedGains()
    thinkplot.plot(bids, gains, label='Player 1')
    print('\nPlayer 1 optimal bid', max(zip(gains, bids)))

    # NOTE: player 2 optimal bid = 31,500, expgain = 19,400, best guess = 40,000
    bids, gains = calc2.expectedGains()
    thinkplot.plot(bids, gains, label='Player 2')
    print('Player 2 optimal bid', max(zip(gains, bids)))

    thinkplot.save(root='price5_expectedGainsFromBids_player1,2',
                   xlabel='bid ($)',
                   ylabel='expected gain ($)',
                   formats=FORMATS)


def plotOptimalBid():
    """Plots optimal bid vs estimated price.
    """
    player1, player2 = makePlayers()
    guesses = numpy.linspace(15000, 60000, 21) # note we don't know the guesses so we make some up.

    res = []
    for guess in guesses:
        player1.makeBeliefs(guess)

        mean = player1.posterior.mean()
        mle = player1.posterior.maximumLikelihood()

        calc = GainCalculator(player1, player2)
        bids, gains = calc.expectedGains()
        gain, bid = max(zip(gains, bids))

        res.append((guess, mean, mle, gain, bid))

    guesses, means, _mles, gains, bids = zip(*res)
    
    thinkplot.prePlot(num=3)
    pyplot.plot([15000, 60000], [15000, 60000], color='gray')
    thinkplot.plot(guesses, means, label='mean')
    #thinkplot.Plot(guesses, mles, label='MLE')
    thinkplot.plot(guesses, bids, label='bid')
    thinkplot.plot(guesses, gains, label='gain')
    thinkplot.save(root='price6_mean_bid_gains',
                   xlabel='guessed price ($)',
                   formats=FORMATS)


def testCode(calc):
    """Check some intermediate results.

    calc: GainCalculator
    """
    # test ProbWin
    for diff in [0, 100, 1000, 10000, 20000]:
        print(diff, calc.probWin(diff))
    print("")

    # test Return
    price = 20000
    for bid in [17000, 18000, 19000, 19500, 19800, 20001]:
        print(bid, calc.gain(bid, price))
    print("")


def main():
    print("---------------------- Plotting expected gains: ----------------------")
    plotExpectedGains()
    print("\n\n\n---------------------- Plotting optimal bid: ----------------------")
    plotOptimalBid()



if __name__ == '__main__':
    main()
