import pandas as pd

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")
obama_text = list(df[df.name=='Barack Obama'].text)[0]
bush_text = list(df[df.name=='George W. Bush'].text)[0]

text = obama_text + ' ' + bush_text
words = [word.lower() for word in text.split()]
words = set(words)

def word_count(text: str, words: set):
    counts = {}

    for word in words:
        counts[word] = 0
        for w in text.split():
            if w.lower() == word:
                counts[word] += 1

    return counts

def find_top_ten(word_count_dict: dict):
    words = list(word_count_dict)
    current_words = words[:10]
    candidates = words[10:]

    for i, word in enumerate(current_words):
        current_word = word
        for w in candidates:
            if word_count_dict[w] > word_count_dict[current_word]:
                current_word = w
                current_words[i] = current_word
        if current_word != word:
            candidates.remove(current_word)

    return {word: word_count_dict[word] for word in current_words}

print(find_top_ten(word_count(obama_text, words)))