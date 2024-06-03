import pandas as pd

data = pd.read_parquet('part-00001-ef7565df-108e-4cd5-9f67-ab7cc7161af2-c000.snappy.parquet')

print(data.columns)

