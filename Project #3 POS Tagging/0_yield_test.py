
## yield statement
mygenerator = (x*x for x in range(3))
print(mygenerator)

for i in mygenerator:
    print(i)

# yield is like return, except the function will return a generator.
def createGenerator():
    mylist = range(3)
    for i in mylist:
        return i*i

def createGenerator():
    mylist = range(3)
    for i in mylist:
        yield i*i

mygenerator = createGenerator()
print(mygenerator)

for i in mygenerator:
    print(i)

