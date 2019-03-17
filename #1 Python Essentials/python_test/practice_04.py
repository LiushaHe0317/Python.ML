
import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(4,3))
df.columns = ['a','b','c']

print(df)
df.a[df.a>.5] = 0
print('\n')
print(np.any(df.isnull()) == True)
