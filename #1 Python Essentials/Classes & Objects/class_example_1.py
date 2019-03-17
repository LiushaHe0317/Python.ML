
class MyCalculator():
     def __init__(self, 
                  name = 'my calculator 1', 
                  price = 10):
          self.name = name
          self.price = price
     def add(self, x, y):
          print(self.name)
          result = x + y
          return result
     def minus(self, x, y):
          print(self.price)
          result = x - y
          return result
     def times(self, x, y):
          result = x*y
          return result
     def division(self, x, y):
          result = x/y
          return result
