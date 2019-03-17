
import re

sentence1 = r'How are you'
sentence2 = r"I'm 30 years old"
sentence3 = r"I was born in the year 1988"

sentence_modified = re.sub("[@+,*'\-]","",sentence2)

print(sentence_modified)