
import spacy

nlp = spacy.load('en_core_web_sm')

def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + '--' + ent.label_ + '--' + str(spacy.explain(ent.label_)))
    else:
        print('No Entities found')

doc = nlp(u"Tesla to build a U.K. factory for $6 million")
show_ents(doc)

# in this situation Tesla may not be recognitzed by spacy
# to deal with this problem
from spacy.tokens import Span
ORG = doc.vocab.strings[u"ORG"]
new_ent = Span(doc, 0, 1, label = ORG)
doc.ents = list(doc.ents) + [new_ent]
show_ents(doc)

## how add multiple named-entities
# let's see an example
doc = nlp(u"our company has created a brand new vacuum cleaner. "
          u"This vacuum cleaner is the best in show.")

show_ents(doc)

# solution
# create a matcher
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)

# create patterns
phrase_list = ['vacuum cleaner', 'vacuum-cleaner']
phrase_patterns = [nlp(text) for text in phrase_list]

# apply patterns to matcher
matcher.add('product', None, *phrase_patterns)
# find each match
found_matches = matcher(doc)

PROD = doc.vocab.strings[u"PRODUCT"]
new_ents = [Span(doc, match[1], match[2], label = PROD) for match in found_matches]

doc.ents = list(doc.ents) + new_ents

show_ents(doc)

doc = nlp(u"I paid $29.95 for this car toys, but now it is marked down to 10 dollars")
find_ents = [ent for ent in doc.ents if ent.label_ == 'MONEY']
print(find_ents)

## visualize the entities
from spacy import displacy

doc = nlp(u"Over the last quarter, Apple sold 20 thousand iPods for a profit of $6 million."
          u"By contrast, Sony only sold 8 thousand walkman music players")

# visualizing
displacy.serve(doc, style = 'ent')
