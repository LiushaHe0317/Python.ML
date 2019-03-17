
import pandas as pd

df = pd.read_csv(r'C:\Users\degere\Desktop\Data Science & ML\#1 Python\My NLP Projects\#3 NLP\#5 POS_NER\NER System Demo\NER_train_data_origin.csv')
df.columns = ['uttrance', 'tag_info']

print(df['uttrance'])
print(df['tag_info'])
