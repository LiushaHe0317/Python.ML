import re
import numpy

initial = {}
second_word = {}
transitions = {}

def remove_punctuation(text: str):
    return re.sub(r'[^\w\s]','',text)

def add2dict(d: dict, k, v):
    if k not in d:
        d[k] = []
    d[k].append(v)

def list2pdict(ts):
    d = {}
    n = len(ts)
    for t in ts:
        d[t] = d.get(t, 0.) + 1
    for t, c in d.items():
        d[t] = c / n
    return d

def sample_word(d: dict):
    p0 = numpy.random.random()
    cumulative = 0
    for t, p in d.items():
        cumulative += p
        if p0 < cumulative:
            return t

def generate(initial, transitions, second_word):
    for i in range(4):
        sentence = []

        w0 = sample_word(initial)
        w1 = sample_word(second_word[w0])

        sentence.append(w0)
        sentence.append(w1)

        while True:
            w2 = sample_word(transitions[(w0, w1)])
            if w2 == 'END':
                break
            sentence.append(w2)

            w0 = w1
            w1 = w2

        print(" ".join(sentence))

file_name = r"../data/robert_frost.txt"

for line in open(file_name):
    tokens = remove_punctuation(line.rstrip().lower()).split()

    t_length = len(tokens)
    for i, t in enumerate(tokens):
        if i == 0:
            initial[t] = initial.get(t, 0.) + 1
        else:
            t_1 = tokens[i-1]
            if i == t_length - 1:
                add2dict(transitions, (t_1, t), 'END')
            if i == 1:
                add2dict(second_word, t_1, t)
            else:
                t_2 = tokens[i-2]
                add2dict(transitions, (t_2, t_1), t)

# normalise the distributions
initial_total = sum(initial.values())
for t, c in initial.items():
    initial[t] = c / initial_total

for t_1, ts in second_word.items():
    second_word[t_1] = list2pdict(ts)

for k, ts in transitions.items():
    transitions[k] = list2pdict(ts)

generate(initial, transitions, second_word)
