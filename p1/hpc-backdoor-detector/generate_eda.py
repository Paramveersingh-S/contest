import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os
import nbformat as nbf

def generate_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # Markdown
    cells.append(nbf.v4.new_markdown_cell("# Exploratory Data Analysis (EDA)\nIn this notebook, we explore the HPC traces dataset."))
    
    # Setup code
    cells.append(nbf.v4.new_code_cell("""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import sys
import os

# Add src to path
sys.path.append('../src')
from data import load_raw_traces

df = load_raw_traces('../../trace.csv')
"""))

    # Basic stats
    cells.append(nbf.v4.new_markdown_cell("## Basic Statistics\nPrint shape, column names, and summary statistics."))
    cells.append(nbf.v4.new_code_cell("""print(f'Shape: {df.shape}')
print('\\nData Types:')
print(df.dtypes)
print('\\nSummary Statistics:')
display(df.describe())
"""))
    
    # Missing values
    cells.append(nbf.v4.new_markdown_cell("## Data Quality\nCheck for missing or invalid values."))
    cells.append(nbf.v4.new_code_cell("""print('Missing values per column:')
print(df.isnull().sum())
print(f'\\nDuplicates: {df.duplicated().sum()}')
"""))
    
    # Distributions
    cells.append(nbf.v4.new_markdown_cell("## Distributions\nHistograms and boxplots for each HPC column."))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(df.columns):
    sns.histplot(df[col], ax=axes[i], kde=True)
    axes[i].set_title(f'Histogram: {col}')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(df.columns):
    sns.boxplot(y=df[col], ax=axes[i])
    axes[i].set_title(f'Boxplot: {col}')
plt.tight_layout()
plt.show()
"""))
    
    # Correlation
    cells.append(nbf.v4.new_markdown_cell("## Correlation\nHeatmap to identify highly correlated features."))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Feature Correlation')
plt.show()
"""))

    # PCA and t-SNE
    cells.append(nbf.v4.new_markdown_cell("## Dimensionality Reduction (PCA & t-SNE)\nVisualize the clean-trace manifold."))
    cells.append(nbf.v4.new_code_cell("""# Standardize first
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f'PCA Explained Variance: {pca.explained_variance_ratio_}')

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
axes[0].set_title('PCA')
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
axes[1].set_title('t-SNE')
plt.show()
"""))

    nb['cells'] = cells
    
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/01_eda.ipynb', 'w') as f:
        nbf.write(nb, f)
        
if __name__ == '__main__':
    generate_notebook()
    print("EDA Notebook generated successfully.")
