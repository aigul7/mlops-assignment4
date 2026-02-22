import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/cancer_reg.csv')
print(f"Original dataset: {len(df)} rows, {len(df.columns)} columns")

# drop geography text column
if 'geography' in df.columns:
    df = df.drop(columns=['geography'])

# drop rows where target is missing
df = df.dropna(subset=['target_deathrate'])

# fill missing values with column median
df = df.fillna(df.median(numeric_only=True))
print(f"After cleaning: {len(df)} rows")

# train/test split (80% train, 20% test)
train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs('data/modified', exist_ok=True)
train.to_csv('data/train_baseline.csv', index=False)
test.to_csv('data/test_baseline.csv', index=False)
print(f"Train size: {len(train)} rows")
print(f"Test size:  {len(test)} rows")

# scenario A: decrease medianincome by 40,000
scenario_a = test.copy()
scenario_a['medianincome'] = scenario_a['medianincome'] - 40000
scenario_a.to_csv('data/modified/scenario_A.csv', index=False)
print(f"\nScenario A created.")
print(f"  medianincome mean: {test['medianincome'].mean():.0f} -> {scenario_a['medianincome'].mean():.0f}")

# scenario AB: A + increase povertypercent by 20
scenario_ab = scenario_a.copy()
scenario_ab['povertypercent'] = scenario_ab['povertypercent'] + 20
scenario_ab.to_csv('data/modified/scenario_AB.csv', index=False)
print(f"\nScenario AB created.")
print(f"  povertypercent mean: {test['povertypercent'].mean():.1f} -> {scenario_ab['povertypercent'].mean():.1f}")

# scenario ABC: AB + increase avghouseholdsize by 2
scenario_abc = scenario_ab.copy()
scenario_abc['avghouseholdsize'] = scenario_abc['avghouseholdsize'] + 2
scenario_abc.to_csv('data/modified/scenario_ABC.csv', index=False)
print(f"\nScenario ABC created.")
print(f"  avghouseholdsize mean: {test['avghouseholdsize'].mean():.2f} -> {scenario_abc['avghouseholdsize'].mean():.2f}")

print("\nAll files created successfully!")
print("Check data/ and data/modified/ folders.")