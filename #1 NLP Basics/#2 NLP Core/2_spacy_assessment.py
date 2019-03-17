
import spacy
nlp = spacy.load('en_core_web_sm')

## requirement
#1 create a doc object from txt file owlcreek.txt

with open('owlcreek.txt') as f:
    doc = nlp(f.read())

#2 answer questions:
#    (1) how many tokens in the file
print(len(doc))

#    (2) how many sentences in the file
#    (3) print the second sentence of the document
num = 0
for sentence in doc.sents:
    num += 1
    if num == 2:
        the_sentence = sentence
        print(the_sentence)
print(num)

# simpler way
#############################################
doc_sentences = [sent for sent in doc.sents]
############################################
print(len(doc_sentences))
print(doc_sentences[1].text)

the_sentence = doc_sentences[1].text

#    (4) For each token in the sentence above, print its text, 
#        POS tag, dep tag and lemma
for token in doc_sentences[1]:
    print(token.text, token.pos_, token.dep_, token.lemma_)

#    (6) Write a matcher called 'Swimming' that finds 
#        both occurrences of the phrase "swimming vigorously" in the text
from spacy.matcher import Matcher
matcher = Matcher(nlp.vocab)

pattern = [{'LOWER':'swimming'},{'IS_SPACE':True, 'OP':'*'},{'LOWER':'vigorously'}]

matcher.add('Swimming', None, pattern)

found_matches = matcher(doc)

#    (7) Print the text surrounding each found match
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]
    span = doc[start - 5:end + 5]
    
    print(span.text)

# print the sentence that contains each found match
for sentence in doc_sentences:
    if found_matches[0][1] < sentence.end:
        print(sentence)
        break

for sentence in doc_sentences:
    if found_matches[1][1] < sentence.end:
        print(sentence)
        break