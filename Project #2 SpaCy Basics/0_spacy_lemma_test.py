
## lemmatization = words reduction + morphological analysis
# find the lemma

import spacy
nlp = spacy.load('en_core_web_sm')

doc1 = nlp(u"I am a runner running in a race because I love to run since I ran today")

for token in doc1:
    print(token.text, token.pos, token.lemma, token.lemma_)
