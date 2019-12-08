import pandas as pd
import numpy

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")
obama_text = df[df.name=='Barack Obama'].text
George_tezt = df[df.name=='George W. Bush'].text
Joe_text = df[df.name=='Joe Biden'].text

all_text = list(obama_text)[0] + ' ' + list(George_tezt)[0] + ' ' + list(Joe_text)[0]
words = [word.lower() for word in all_text.split()]
words = set(words)

def word_vector(text: str, words: set):
    vec = []
    for word in words:
        count = 0
        for w in text.split():
            if w.lower() == word:
                count += 1
        vec.append(count)
    return numpy.array(vec)

o_vec = word_vector(list(obama_text)[0], words)
g_vec = word_vector(list(George_tezt)[0], words)
j_vec = word_vector(list(Joe_text)[0], words)

og_sim = numpy.linalg.norm(o_vec-g_vec)
oj_sim = numpy.linalg.norm(o_vec-j_vec)
gj_sim = numpy.linalg.norm(g_vec-j_vec)

print('A. ', oj_sim)
print('B. ', og_sim)
print('C. ', gj_sim)