
class Calculator:

    name = 'Liusha'

    price = 18

    def add(self, x, y):
        result = x + y
        print(result)

    def minus(self, x, y):
        result = x - y
        print(result)

    def times(self, x, y):
        print(x*y)

    def devide(self, x, y):
        print(x/y)

    def power(self, x, y):
        print(x**y)


c = Calculator()

c.power(4, 5)
print(c)

for i in [1, 2, 3, 4]:
    print(i)
