import pandas as pd 
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/metadata.csv", sep = "|", names=["file","text","speaker"])

train, val = train_test_split(
    df,
    test_size= 0.05,
    random_state= 42,
    stratify= df["speaker"]
)

train.to_csv("data/train_metadata.csv", sep= "|", index = False, header = False)
val.to_csv("data/val_metadata.csv", sep = "|", index = False, header = False )

print(f" Train: {len(train)} samples \n Val: {len(val)} samples")
