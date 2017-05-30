import Pmf
#arg types: (t, name='')
pmf = Pmf.MakePmfFromList([1, 2, 2, 3, 5])
"""print(pmf)"""

print(pmf.Prob(2))
#>>> 0.4

print(pmf.Incr(2, 0.2)) #args: number value, how much to increment by
print(pmf.Prob(2))
#>>> 0.6

print(pmf.Mult(2, 0.5))
print(pmf.Prob(2))
#>>>0.3

#call Total to return sum of probabilities.
# If sum != 1, probablilites are no longer normalized.
print(pmf.Total())
#>>> 0.9

#call Normalize to renormalize:
print(pmf.Normalize())
print(pmf.Total())
#>>> 1.0

