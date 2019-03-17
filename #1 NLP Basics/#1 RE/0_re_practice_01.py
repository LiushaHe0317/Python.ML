
import re

sentence = 'my cell phone number is 759-642-7127'

## match function
print(re.match(r'.',sentence))
print(re.match(r'.*',sentence))
print(re.match(r'.+',sentence))
print('\n')

## search function
# search for texts
pattern1 = r'phone'
my_match1 = re.search(pattern1, sentence)
match1_span = my_match1.span()

# search for digits
pattern2 = r'\d{3}-\d{3}-\d{4}'
my_match2 = re.search(pattern2, sentence)

print('Text spans from '+str(match1_span[0])+' to '+str(match1_span[1]))
print('The number is '+str(my_match2.group()))
print('\n')

## sub function
my_modified1 = re.sub(r'\d{3}-\d{3}-\d{4}','',sentence)
print(my_modified1)

my_modified2 = re.sub(r'\d','#',sentence, flags = re.I)
print(my_modified2)