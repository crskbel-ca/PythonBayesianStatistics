import Pmf
import matplotlib.pyplot as pyplot

hist = Pmf.MakeHistFromList([1, 2, 2, 3, 5]) #arg types: (t, name ''

pyplot.pie([1, 2, 3])
pyplot.show()

vals, freqs = hist.Render()
rectangles = pyplot.bar(vals, freqs) # values on x-axis, freqs on y-axis
pyplot.show()