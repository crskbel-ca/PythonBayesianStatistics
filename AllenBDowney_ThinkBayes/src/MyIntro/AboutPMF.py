


from src.thinkbayes import PMF

# simple setting example, no need for normalize()
pmfNum = PMF()
for num in [1,2,3,4,5,6]:
    pmfNum.set(num, 1/6.0)

pmfNum.printSuite()



# normalizing example
wordList = ["the", "bright", "red", "fox", "jumped", "over", "the", "lazy", "dog",
            "and", "the", "fox", "scurried", "up", "the", "tree", "before", "the",
            "dog", "could", "find", "him"]
pmfWord = PMF()
for word in wordList:
    pmfWord.incr(word, 1) # increasing count of each word

print("\nPrinting before normalizing: ")
pmfWord.printSuite()
print("\nPrinting after normalizing: ")
pmfWord.normalize()
pmfWord.printSuite()


print("\n\nprob(the) = ", pmfWord.prob("the"))
print("prob(fox) = ", pmfWord.prob("fox"))