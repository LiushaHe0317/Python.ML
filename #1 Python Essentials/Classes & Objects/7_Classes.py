
class SimpleClass:
    greeting ="Hello"
    def funct1(self):
        print("This is a simple class")

simple_object = SimpleClass()

class Computer:
    name  = ""
    kind  = "Laptop"
    color = ""
    cost  = 500
    def description(self):
        desc_str = "%s is a %s %s worth $%.2f."%(self.name,self.color,self.kind,self.cost)
        return desc_str
