import pandas as pd
import itertools

data=pd.read_excel(r'C:\Users\admin\Desktop\源.xlsx',usecols=[0]).values.tolist()
data=list(itertools.chain.from_iterable(data))
print(data)