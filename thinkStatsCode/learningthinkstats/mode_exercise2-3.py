import Pmf

hist = Pmf.MakeHistFromList([1, 2, 2, 3, 5])


#function mode takes Hist object and returns most frequent value
def mode():
    for val in sorted(hist.Values()):
        return val, hist.Freq(val)

print(mode())


### SOLVE BUG: PROGRAM DOES NOT RETURN MOST COMMON VALUE IF 1 starts the list.
