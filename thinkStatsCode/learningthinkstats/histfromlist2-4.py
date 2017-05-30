import Pmf


hist = Pmf.MakeHistFromList([1, 2, 2, 3, 5])
print(hist)

print("To loop through values in order, use function sorted:")
for val in sorted(hist.Values()):
    print(val, hist.Freq(val))

print("Function Items returns unsorted list of value-freq pairs:")
for val, freq in hist.Items():
    print(val, freq)


