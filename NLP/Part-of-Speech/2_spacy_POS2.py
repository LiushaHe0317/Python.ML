
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy import displacy

# 1. Create a Doc object from the file peterrabbit.txt
with open('peterrabbit.txt') as f:
    doc = nlp(f.read())

doc_sentences = [sent for sent in doc.sents]

for token in doc_sentences[2]:
    print(token.text, token.pos_, token.tag_, str(spacy.explain(token.tag_)))

# 3. Provide a frequency list of POS tags from the entire document
# 4. CHALLENGE: What percentage of tokens are nouns?
pos_dic = doc.count_by(spacy.attrs.POS)
num = 0
for k, v in sorted(pos_dic.items()):
    print(k, doc.vocab[k].text, v)
    num = num + v
    if doc.vocab[k] == 'NOUN':
        noun_num = v

noun_percent = noun_num/ num
print(noun_percent)

# 5. Display the Dependency Parse for the third sentence
displacy.serve(doc_sentences[2], style = 'dep')

# *6. Show the first two named entities from Beatrix Potter's *The Tale of Peter Rabbit **
mystring = u"Beatrix Potter's *The Tale of Peter Rabbit **"
def show_ents(doc):
    if doc.ents:
        for ent in doc.ents:
            print(ent.text + '--' + ent.label_ + '--' + str(spacy.explain(ent.label_)))

doc1 = nlp( mystring )
show_ents(doc1)

# 7. How many sentences are contained in The Tale of Peter Rabbit?
print(len(doc_sentences))

# 8. CHALLENGE: How many sentences contain named entities?
list_of_sents = [nlp(sent.text) for sent in doc.sents]
list_of_ners = [doc for doc in list_of_sents if doc.ents]
print(len(list_of_ners))

# 9. CHALLENGE: Display the named entity visualization 
#    for list_of_sents[0] from the previous problem
displacy.serve(list_of_sents[0], style = 'ent')
