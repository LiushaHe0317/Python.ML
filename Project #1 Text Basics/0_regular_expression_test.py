
import re

text = 'my cell phone number is 759-642-7127'

# search for texts
pattern1 = 'phone'
my_match = re.search(pattern1, text)
my_match.span()

# search for digit number
pattern2 = r'\d{3}-\d{3}-\d{4}'
result = re.search(pattern2, text)

# show the result
the number = result.group()