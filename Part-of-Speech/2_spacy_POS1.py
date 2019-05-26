
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
# see the tag
for token in doc:
    print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))

doc = nlp(u"I read books on NLP.")
for token in doc:
    print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))

doc = nlp(u"I read a book on NLP.")
for token in doc:
    print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))

doc = nlp(u"I've read a book on NLP.")
for token in doc:
    print(token.text, token.pos_, token.tag_, spacy.explain(token.tag_))

# pos number dictionary
doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")
pos_dic = doc.count_by(spacy.attrs.POS)

# do some test
print(doc.vocab[83].text)
print(doc[2].pos)

for k,v in sorted(pos_dic.items()):
    print(k, doc.vocab[k].text, v)

# tag number dic
tag_dic = doc.count_by(spacy.attrs.TAG)

for k,v in sorted(tag_dic.items()):
    print(k, doc.vocab[k].text, v)

# dep number dic
dep_dic = doc.count_by(spacy.attrs.DEP)
for k,v in sorted(dep_dic.items()):
    print(k, doc.vocab[k].text, v)











