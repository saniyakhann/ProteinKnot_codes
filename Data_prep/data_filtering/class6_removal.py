import pandas as pd

input  = "single_domain_proteins_complete.csv"
output = "proteins_final.csv"

df = pd.read_csv(input)
before = len(df)

#class = first component of cath_code
df["class"] = df["cath_code"].str.split(".").str[0].astype(int)

#keep only Classes 1-4 
df_filtered = df[df["class"].isin([1, 2, 3, 4])].drop(columns=["class"])
after = len(df_filtered)

df_filtered.to_csv(output, index=False)

print(f"Before:  {before}")
print(f"After:   {after}")
