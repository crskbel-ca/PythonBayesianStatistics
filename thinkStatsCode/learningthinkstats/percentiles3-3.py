
scores = [55, 66, 77, 88, 99]
yourScore = 88


def PercentileRank(scores, yourScore):
    count = 0
    for score in scores:
        if score <= yourScore:
            count += 1
    percentileRank = 100.0 * count / len(scores)
    return percentileRank


def Percentile(scores, percentileRank):
    scores.sort()
    for score in scores:
        if PercentileRank(scores, score) >= percentileRank:
            return score


resultPRank = PercentileRank(scores, yourScore)
print(Percentile(scores, resultPRank))