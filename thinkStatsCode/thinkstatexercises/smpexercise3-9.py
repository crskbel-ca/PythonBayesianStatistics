import random
#import Pmf
import bisect



def Value(self, p):
    """Returns InverseCDF(p), the value that corresponds to probability p.

    Args:
        p: number in the range [0, 1]

    Returns:
        number value
    """
    if p < 0 or p > 1:
        raise ValueError('Probability p must be in range [0, 1]')

    if p == 0:
        return self.xs[0]
    if p == 1:
        return self.xs[-1]

    index = bisect.bisect(self.ps, p)

    if p == self.ps[index-1]:
        return self.xs[index-1]
    else:
        return self.xs[index]


def Random(self):
    """Chooses a random value from this distribution."""
    return self.Value(random.random())


def Sample(self, n):
    """Generates a random sample from this distribution.

    Args:
        n: int length of the sample
    """
    print("hello")
    return [self.Random() for i in range(n)]



n = 5
result = Sample(n)
print(result)