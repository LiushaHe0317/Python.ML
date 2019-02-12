
import spacy

# loading language processing model
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"Tesla is looking at buying US startup for $6 million")
doc2 = nlp(u"Tesla isn't looking into startups anymore.")
doc3 = nlp(u"This is the first sentence. This is another sentence. This is the last sentence.")

for token in doc:
    print(token.text, token.pos, token.pos_, token.dep_)

for token in doc2:
    print(token.text, token.pos, token.pos_, token.dep_)

for sentence in doc3.sents:
    print(sentence)
