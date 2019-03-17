
# create a txt file
with open('test.txt', 'w') as f:
    f.write('Hello World\n')
    f.write('this is the second iine')

with open('test.txt', 'r') as f:
   content = f.readlines()
   for each_line in content:
       sentence = each_line.split()
       print(sentence)

with open('test.txt', 'a') as f:
    f.write('\nThis is the third line\n')
    
