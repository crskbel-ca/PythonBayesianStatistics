"""This file contains code for use with "Think Bayes",
by Allen B. Downey, available from greenteapress.com

Copyright 2012 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from src.thinkbayes import PMF


pmf = PMF()

# Setting the Prior distribution: This is p(B1) and p(B2)
pmf.set('Bowl 1', 0.5)
pmf.set('Bowl 2', 0.5)

# Multiplying priors by corresponding likelihood: This is p(B1) * p(V|B1)
pmf.mult('Bowl 1', 0.75)
pmf.mult('Bowl 2', 0.5)

pmf.normalize() # allowed, because the hypotheses are mutually exclusive + collectively exhaustive.

# Result is the Posterior distribution.
print("p(B1 | V) = ", pmf.prob('Bowl 1')) # getting posterior distributions for each hypothesis
print("p(B2 | V) = ", pmf.prob('Bowl 2'))
