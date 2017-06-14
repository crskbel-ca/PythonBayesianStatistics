"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import math

import thinkbayes
import thinkplot
import thinkstats

from src import columns

USE_SUMMARY_DATA = True

class Hockey(thinkbayes.Suite):
    """Represents hypotheses about the scoring rate for a team. """

    def __init__(self, name=''):
        """Initializes the Hockey object.

        name: string
        """
        if USE_SUMMARY_DATA:
            # prior based on each team's average goals scored
            mu = 2.8
            sigma = 0.3
        else:
            # prior based on each pair-wise match-up
            mu = 2.8
            sigma = 0.85

        pmf = thinkbayes.makeGaussianPmf(mu, sigma, 4)
        thinkbayes.Suite.__init__(self, pmf, name=name)

    def likelihood(self, data, hypo):
        """Computes the likelihood of the data under the hypothesis.

        Evaluates the Poisson PMF for lambda and k.

        hypo: goal scoring rate in goals per game
        data: goals scored in one period
        """
        lmbda = hypo # note: the goal scoring average
        k = data # note: the amount of goals scored in one time period.
        like = thinkbayes.evalPoissonPmf(k, lmbda)
        return like


def makeGoalPmf(suite, high=10):
    """Makes the distribution of goals scored, given distribution of lam.

    suite: distribution of goal-scoring rate
    high: upper bound

    returns: Pmf of goals per game
    """
    metaPmf = thinkbayes.PMF()

    for lam, prob in suite.items():
        pmf = thinkbayes.makePoissonPmf(lam, high)
        metaPmf.set(pmf, prob)

    mix = thinkbayes.makeMixture(metaPmf, name=suite.name)
    return mix


def makeGoalTimePmf(suite):
    """Makes the distribution of time til first goal.

    suite: distribution of goal-scoring rate

    returns: Pmf of goals per game
    """
    metaPmf = thinkbayes.PMF()

    for lam, prob in suite.items():
        pmf = thinkbayes.makeExponentialPmf(lam, high=2, n=2001)
        metaPmf.set(pmf, prob)

    mix = thinkbayes.makeMixture(metaPmf, name=suite.name)
    return mix


class Game(object):
    """Represents a game.

    Attributes are set in columns.read_csv.
    """
    convert = dict()

    def clean(self):
        self.goals = self.pd1 + self.pd2 + self.pd3


def readHockeyData(filename='hockey_data.csv'):
    """Read game scores from the data file.

    filename: string
    """
    game_list = columns.read_csv(filename, Game)

    # map from gameID to list of two games
    games = {}
    for game in game_list:
        if game.season != 2011:
            continue
        key = game.game
        games.setdefault(key, []).append(game)

    # map from (team1, team2) to (score1, score2)
    pairs = {}
    for key, pair in iter(games):
        t1, t2 = pair
        key = t1.team, t2.team
        entry = t1.total, t2.total
        pairs.setdefault(key, []).append(entry)

    processScoresTeamwise(pairs)
    processScoresPairwise(pairs)


def processScoresPairwise(pairs):
    """Average number of goals for each team against each opponent.

    pairs: map from (team1, team2) to (score1, score2)
    """
    # map from (team1, team2) to list of goals scored
    goals_scored = {}
    for key, entries in pairs.iteritems():
        t1, t2 = key
        for entry in entries:
            g1, g2 = entry
            goals_scored.setdefault((t1, t2), []).append(g1)
            goals_scored.setdefault((t2, t1), []).append(g2)

    # make a list of average goals scored
    lams = []
    for key, goals in iter(goals_scored):
        if len(goals) < 3:
            continue
        lam = thinkstats.mean(goals)
        lams.append(lam)

    # make the distribution of average goals scored
    cdf = thinkbayes.makeCdfFromList(lams)
    thinkplot.cdf(cdf)
    thinkplot.show()

    mu, var = thinkstats.meanAndVariance(lams)
    print('mu, sig', mu, math.sqrt(var))

    print('BOS v VAN', pairs['BOS', 'VAN'])


def processScoresTeamwise(pairs):
    """Average number of goals for each team.

    pairs: map from (team1, team2) to (score1, score2)
    """
    # map from team to list of goals scored
    goals_scored = {}
    for key, entries in pairs.iteritems():
        t1, t2 = key
        for entry in entries:
            g1, g2 = entry
            goals_scored.setdefault(t1, []).append(g1)
            goals_scored.setdefault(t2, []).append(g2)

    # make a list of average goals scored
    lams = []
    for key, goals in iter(goals_scored):
        lam = thinkstats.mean(goals)
        lams.append(lam)

    # make the distribution of average goals scored
    cdf = thinkbayes.makeCdfFromList(lams)
    thinkplot.cdf(cdf)
    thinkplot.show()

    mu, var = thinkstats.meanAndVariance(lams)
    print('mu, sig', mu, math.sqrt(var))


def main():
    #ReadHockeyData()
    #return

    formats = ['pdf', 'eps']

    suite1 = Hockey('bruins')
    suite2 = Hockey('canucks')

    thinkplot.clf()
    thinkplot.prePlot(num=2)
    thinkplot.pmf(suite1)
    thinkplot.pmf(suite2)
    thinkplot.save(root='hockey0',
                   xlabel='Goals per game',
                   ylabel='Probability',
                   formats=formats)

    suite1.updateSet([0, 2, 8, 4])
    suite2.updateSet([1, 3, 1, 0])

    thinkplot.clf()
    thinkplot.prePlot(num=2)
    thinkplot.pmf(suite1)
    thinkplot.pmf(suite2)
    thinkplot.save(root='hockey1',
                   xlabel='Goals per game',
                   ylabel='Probability',
                   formats=formats)
    # NOTE: most likley lambda = 2.6 for canucks and 2.9 for bruins

    # NOTE: for each lam, there is poisson PMF. Now combine in mixture to create meta pmf
    # NOTE: Each pmf is weighted to probabilities in distribution of lambda.
    goal_dist1 = makeGoalPmf(suite1)
    goal_dist2 = makeGoalPmf(suite2)

    thinkplot.clf()
    thinkplot.prePlot(num=2)
    thinkplot.pmf(goal_dist1)
    thinkplot.pmf(goal_dist2)
    thinkplot.save(root='hockey2',
                   xlabel='Goals',
                   ylabel='Probability',
                   formats=formats)

    # NOTE: for each lambda, there is exponential PMF. Combine in mixture, as above
    time_dist1 = makeGoalTimePmf(suite1)
    time_dist2 = makeGoalTimePmf(suite2)

    print('MLE bruins', suite1.maximumLikelihood())
    print('MLE canucks', suite2.maximumLikelihood())

    thinkplot.clf()
    thinkplot.prePlot(num=2)
    thinkplot.pmf(time_dist1)
    thinkplot.pmf(time_dist2)
    thinkplot.save(root='hockey3',
                   xlabel='Games until goal',
                   ylabel='Probability',
                   formats=formats)


    # NOTE: POISSON: Goal pmf calculations: lambda poissons mixture
    diff = goal_dist1 - goal_dist2 # note: the goal differential: if pos, bruins win, if neg, canucks, if 0, tie
    p_win = diff.probGreater(0)
    p_loss = diff.probLess(0)
    p_tie = diff.prob(0)

    print(p_win, p_loss, p_tie)

    # NOTE: EXPONENTIAL: time between goals calculations:
    p_overtime = thinkbayes.pmfProbLess(time_dist1, time_dist2)
    p_adjust = thinkbayes.pmfProbEqual(time_dist1, time_dist2)
    p_overtime += p_adjust / 2
    print('p_overtime', p_overtime )

    print(p_overtime * p_tie)

    # NOTE: total prob(win next game) = probGreater(0) + prob(overtime AND tie) (because to be overtime you have to tie)
    p_win += p_overtime * p_tie
    print('p_win', p_win)

    # key: to win the series, they can either: option 1: win next two, tie the third,
    # or option2: split next two, win the third.
    # NOTE: option 1 - win the next two
    p_series = p_win**2

    # NOTE: option 2 - split the next two, win the third
    p_series += 2 * p_win * (1-p_win) * p_win

    print('p_series', p_series)


if __name__ == '__main__':
    main()
