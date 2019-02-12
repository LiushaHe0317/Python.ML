
import spacy
nlp = spacy.load('en_core_web_sm')

doc = nlp(u"This is a sentence. This is another sentence. This is third sentence.")

doc_sentences = [sent for sent in doc.sents]
print(doc_sentences[1].text)

doc = nlp(u'"Management is doing the right things; leadership is doing the right things." -Peter Drucker')

# add a segmentation rule
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == ';':
            doc[token.i+1].is_sent_start = True
    return doc
# set the nlp model
nlp.add_pipe(set_custom_boundaries, before = 'parser')

# test the model
doc_sentences = [sent for sent in doc.sents]
print(doc_sentences[1])

# change segmentation rules
from spacy.pipeline import SentenceSegmenter

mystring = u"This is a sentence. This is another. \n\n This is a \n third sentence."

def change_boundaries(doc):
    start = 0
    seen_newline = False
    
    for token in doc:
        if seen_newline:
            yield doc[start:token.i]
            start = token.i
            seen_newline = False
        elif token.text.startswith('\n'):
            seen_newline = True
    yield doc[start:]

sbd = SentenceSegmenter(nlp.vocab, strategy = change_boundaries)
nlp.add_pipe(sbd)

doc = nlp(mystring)

for sentence in doc.sents:
    print(sentence)