
import spacy

nlp = spacy.load('en_core_web_sm')

print(nlp.Defaults.stop_words)
len(nlp.Defaults.stop_words)

## Add a stop word
nlp.Defaults.stop_words.add('btw')
nlp.vocab['btw'].is_stop = True
len(nlp.Defaults.stop_words)

## remove a stop word
nlp.Defaults.stop_words.remove('beyond')
nlp.vocab['beyong'].is_stop = False
len(nlp.Defaults.stop_words)

## test if a given word is a stop word
nlp.vocab['beyong'].is_stop
