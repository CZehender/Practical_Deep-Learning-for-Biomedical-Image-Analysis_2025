import pandas as pd

# Load full dataset
df = pd.read_csv("DEN_OG.csv")

# Filter based on path content
df_trainval = df[df['image'].str.contains("train_set|val_set", na=False)]
df_test     = df[df['image'].str.contains("test_set", na=False)]

# Save to new CSVs
df_trainval.to_csv("DEN.csv", index=False)       # overwrite with just train/val
df_test.to_csv("DEN_VAL.csv", index=False)       # new file for validation
