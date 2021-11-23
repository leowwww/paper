import pandas as pd
import numpy as np
import scipy.io as scio
df = scio.loadmat('DATA_AU_TDX.mat')['data']
print(df)
df = pd.DataFrame(df)
df.to_excel('DATA_AU_TDX.xlsx')