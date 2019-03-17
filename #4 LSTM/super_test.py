
class A():
    def __init_(self):
        print('First')

class B():
    def __init__(self):
        print('Second')

class C(A,B):
    def __init__(self):
        super().__init__()
        print('\n')
        print('Third')

py_object = C()