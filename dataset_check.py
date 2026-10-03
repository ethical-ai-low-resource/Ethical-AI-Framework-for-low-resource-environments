import pandas as pd

df = pd.read_excel(r"C:\Users\Faiz Computers\Downloads\50KFinal_urduDataset.xlsx")

sample_df = df.sample(n=5000, random_state=42)

sample_df.to_excel("urdu_5k_dataset.xlsx", index=False)

print(sample_df.shape)