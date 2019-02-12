
import spacy
from spacy.matcher import Matcher

nlp = spacy.load('en_core_web_sm')

## Vocabulary matcher
## create oatterns
matcher = Matcher(nlp.vocab)

pattern1 = [{'LOWER': 'solorpower'}]
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)
doc = nlp(u"The Solar Power industry continues to grow as solar power increases. Solar-power is amazing")

found_matches = matcher(doc)
print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)

## chang pattern
matcher.remove('SolarPower')

pattern1 = [{'LOWER': 'solarpower'}]
pattern2 = [{'LOWER': 'solar'},{'IS_PUNCT': True, 'OP': '*'},{'LOWER': 'power'}]

matcher.add('solarpower', None, pattern1, pattern2)

doc = nlp(u"Solar--power is solorpower yay!")

found_matches = matcher(doc)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start:end]
    print(match_id, string_id, start, end, span.text)

## phrase matching
from spacy.matcher import PhraseMatcher

matcher2 = PhraseMatcher(nlp.vocab)

with open('reaganomics.txt') as f:
    doc3 = nlp(f.read())

phrase_list = ['voodoo economics', 
               'supply-side economics', 
               'trickle-down economics', 
               'free-market economics']
phrase_patterns = [nlp(text) for text in phrase_list]

matcher2.add('EconMarket', None, *phrase_patterns)

found_matches = matcher2(doc3)
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc3[start:end]
    print(match_id, string_id, start, end, span.text)