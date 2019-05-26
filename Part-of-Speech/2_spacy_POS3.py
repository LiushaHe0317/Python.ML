
import spacy
from spacy import displacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"The quick brown fox jumped over the lazy dog's back.")

options = {'distance':60}
displacy.serve(doc, style = 'dep', option = options)

doc2 = nlp(u"This is a sentence. This is another sentence, possibly longer than the previous one.")
spans = list(doc2.sents)
displacy.serve(spans, style = 'dep', options = options)
