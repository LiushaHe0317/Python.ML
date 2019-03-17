
a = ['a','b','e','y','m']
b = [10,5,6,9,3]

dict1 = {}
dict2 = {}
for i in range(len(a)):
    dict1[a[i]] = b[i]
    dict2[b[i]] = a[i]

print(dict2.get(10))
