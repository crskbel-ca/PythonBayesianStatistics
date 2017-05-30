"""import survey
table = survey.Pregnancies()
table.ReadRecords()
print('Number of pregnancies', len(table.records))


r1.setname("Ana")
r2.setname("Nick")
"""


t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

hist = {}
for x in t:
    hist[x] = hist.get(x, 0) + 1
    print("x = ", x)
    print("hist[x] = ", hist[x])

n = float(len(t))
pmf = {}
for x, freq in hist.items():
    pmf[x] = freq / n
    print("x =", x)
    print("freq = ", freq)
    print("n = ", n)
    print("pmf[x] = ", pmf[x])