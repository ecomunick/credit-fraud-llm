import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load raw for EDA
df = pd.read_csv("data/raw/creditcard.csv")

# Feature importance through correlation
plt.figure(figsize=(12, 10))
correlations = df.corr()['Class'].drop('Class').sort_values()
correlations.plot(kind='barh', color='salmon')
plt.title('Correlation with Target (Class)')
plt.xlabel('Correlation Coefficient')
os.makedirs('experiments', exist_ok=True)
plt.savefig('experiments/feature_correlation.png', bbox_inches='tight')

print("Correlation analysis saved to experiments/feature_correlation.png")
