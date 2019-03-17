
## yield statement
mygenerator1 = [x*x for x in range(3)]
print('g1', mygenerator1)

# yield is like return, except the function will return a generator.
def createGenerator2():
    mylist = range(3)
    for i in mylist:
        return i*i
def createGenerator3():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator2 = createGenerator2()
mygenerator3 = createGenerator3()

print('g2', mygenerator2)
print('g3', mygenerator3)
