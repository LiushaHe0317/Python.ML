
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

mystring = '"we\'re moving to L.A.!"'
print(mystring)

doc = nlp(mystring)

for token in doc:
    print(token.text)

doc2 = nlp(u"we're here to help! Send snail-mail, email support@oursite.com or visit us at http:\\www.oursite.com")

for t in doc2:
    print(t)
    
doc3 = nlp(u"A 5km eide costs $10.30")

for t in doc3:
    print(t)

# recognitze entities
doc3 = nlp(u'Apple to build Hong Kong factory in $6 million')

for t in doc3.ents:
    print(t)
    print(t.label_)
    print(str(spacy.explain(t.label_)))
    print('\n')

# recognize chunk
doc4 = nlp(u'Autonomous cars shift insurance liabilities toward manufacturer.')

for chunk in doc4.noun_chunks:
    print(chunk.text, end = ' | ')
    
doc5 = nlp(u'Apple is going to build a U.K. factory for $6 milion')
displacy.serve(doc5, style = 'dep', options = {'distance':110})



