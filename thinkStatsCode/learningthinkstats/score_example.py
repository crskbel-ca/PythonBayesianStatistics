"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""


scores = [55, 66, 77, 88, 99]
yourScore = 88


def PercentileRank(scores, yourScore):
    """Computes the percentile rank relative to a sample of scores."""
    count = 0
    for score in scores:
        if score <= yourScore:
            count += 1
    percentile_rank = 100.0 * count / len(scores)
    print("yourScore:", yourScore)
    return percentile_rank


print('score, percentile rank')
for score in scores:
    print(score, PercentileRank(scores, score))
print("")

def Percentile(scores, percentile_rank):
    """Computes the value that corresponds to a given percentile rank. """
    scores.sort()
    for score in scores:
        if PercentileRank(scores, score) >= percentile_rank:
            return score

def Percentile2(scores, percentile_rank):
    """Computes the value that corresponds to a given percentile rank.
    Slightly more efficient.
    """
    scores.sort()
    index = percentile_rank * (len(scores)-1) / 100
    #convert to int
    return scores[int(index)]


print('prank, score, score')
for percentile_rank in [0, 20, 25, 40, 50, 60, 75, 80, 100]:
    print(percentile_rank, end=' ')
    print(Percentile(scores, percentile_rank), end=' ')
    print(Percentile2(scores, percentile_rank))

