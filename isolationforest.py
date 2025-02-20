from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import pyfastx
import glob
import os
import numpy as np
import pandas as pd
import requests
csv = "cry1realvariations (1).csv"
df = pd.read_csv(csv, usecols=['chromEnd', 'ref', 'alt', 'AF', 'genes', 'variation_type', '_displayName'])

print(df.head())
print(df.columns)

for index, row in df.iterrows():
            gnomAD_ID = row["chromEnd"]
            alternate = row["alt"]
            reference = row["ref"]
            display = row["_displayName"]
            print(display)
            print(alternate)
            print(reference)
            print(gnomAD_ID)
