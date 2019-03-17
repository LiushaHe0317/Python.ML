
import pickle

dict = {'a':[1,2,3,4],'b':[4,5,6,7],'c':1,'d':{1,2,3,4}}

with open('pickle_example.pickle','wb') as file:
    pickle.dump(dict,file)

with open('pickle_example.pickle', 'rb') as file:
    dict1 = pickle.load(file)

print(dict1)