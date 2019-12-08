import pandas as pd

df = pd.read_csv(r"C:\Users\heliu\Desktop\ML Course\Clustering\W2_1\people_wiki.csv")
obama = df[df.name=='Barack Obama'].index.values[0]

print(obama)