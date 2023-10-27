import numpy as np
import pandas as pd

DATA_PATH = "./data/mbti_i.csv"
df = pd.read_csv(DATA_PATH)
df.head()

mbti_types = np.unique(np.array(df['type']))