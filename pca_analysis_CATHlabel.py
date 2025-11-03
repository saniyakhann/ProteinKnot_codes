#What I want from this code: 
# Perform PCA create visualisations for: 
    #1. Variance and compoenents 
    #2. PCA scatter plot
   # 3. Feature correlation heatmap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

plt.style.use('seaborn-v0_8-darkgrid')

features_df = pd.read_csv('protein_features.csv') #feature vector data (only 1000 proteins calculated)
cath_df = pd.read_csv('CATH_proteins.csv') #CATH labelled data 

def prepare_features_with_cath(features_df, cath_df):
    cath_df = cath_df.rename(columns={'protein_id': 'domain_id'})
    merged_df = features_df.merge(cath_df, on='domain_id', how='left')
    
    #extract all CATH hierarchy levels
    def extract_cath_level(code, level):
        if pd.isna(code) or str(code) == 'nan':
            return 'No CATH'
        parts = str(code).split('.')
        if level == 'C':
            return parts[0] if len(parts) >= 1 else 'No CATH'
        elif level == 'A':
            return '.'.join(parts[:2]) if len(parts) >= 2 else 'No CATH'
        elif level == 'T':
            return '.'.join(parts[:3]) if len(parts) >= 3 else 'No CATH'
        elif level == 'H':
            return '.'.join(parts[:4]) if len(parts) >= 4 else 'No CATH'
    
    merged_df['cath_C'] = merged_df['cath_code'].apply(lambda x: extract_cath_level(x, 'C'))
    merged_df['cath_A'] = merged_df['cath_code'].apply(lambda x: extract_cath_level(x, 'A'))
    merged_df['cath_T'] = merged_df['cath_code'].apply(lambda x: extract_cath_level(x, 'T'))
    merged_df['cath_H'] = merged_df['cath_code'].apply(lambda x: extract_cath_level(x, 'H'))

    #features except for size 
    feature_cols = [
        'writhe_mean', 'writhe_std', 
        'writhe_min', 'writhe_max', 'writhe_range', 'writhe_skew',
        'writhe_kurtosis', 'diagonal_mean', 'off_diag_mean', 
        'abs_mean', 'entropy', 'norm_mean', 'complexity'
    ]
    
    X = merged_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, feature_cols, merged_df


def plot_pca_by_cath_level(X_pca, merged_df, cath_column, level_name, pca, ax): 
    #get unique categories
    categories = sorted([c for c in merged_df[cath_column].unique() if c != 'No CATH'])
    n_categories = len(categories)
    
    colors = plt.cm.tab20(np.linspace(0, 1, len(categories)))
    color_dict = dict(zip(categories, colors))
    
    #plot each category
    for category in categories:
        mask = merged_df[cath_column] == category
        if mask.sum() > 0:
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                      c=[color_dict[category]], 
                      label=f'{category}', alpha=0.6, s=40, 
                      edgecolors='black', linewidths=0.3)

    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)', fontsize=10)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)', fontsize=10)
    ax.set_title(f'CATH {level_name}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    #legend
    if len(categories) <= 10:
        ax.legend(fontsize=7, loc='best', ncol=1)
    else:
        ax.legend(fontsize=6, loc='best', ncol=2)


def pca_analysis_all_levels(X_scaled, feature_cols, merged_df):  
    #perform PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    #print statistics
    print("PCA varience:")
    print(f"PC1: {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"PC2: {pca.explained_variance_ratio_[1]*100:.2f}%")
    print(f"Total: {sum(pca.explained_variance_ratio_)*100:.2f}%")
    print()
    
    #statistics for each CATH level
    print("CATH stats:")
    for level, col in [('Class (C)', 'cath_C'), 
                       ('Architecture (A)', 'cath_A'), 
                       ('Topology (T)', 'cath_T'),
                       ('Homology (H)', 'cath_H')]:
        n_unique = merged_df[col].nunique() - (1 if 'No CATH' in merged_df[col].values else 0)
        print(f" Unique categories: {n_unique}")
    print()
    
    #component loading
    loadings_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=feature_cols
    )
    
    print("\nPC1 - Top 5:")
    for feature in loadings_df['PC1'].abs().sort_values(ascending=False).head(5).index:
        print(f"  {feature:20s}: {loadings_df.loc[feature, 'PC1']:+.3f}")
    
    print("\nPC2 - Top 5:")
    for feature in loadings_df['PC2'].abs().sort_values(ascending=False).head(5).index:
        print(f"  {feature:20s}: {loadings_df.loc[feature, 'PC2']:+.3f}")
    print()
    
    #create 2x2 subplot for all 4 CATH levels
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    plot_pca_by_cath_level(X_pca, merged_df, 'cath_C', 'Class (C)', pca, axes[0, 0])
    plot_pca_by_cath_level(X_pca, merged_df, 'cath_A', 'Architecture (C.A)', pca, axes[0, 1])
    plot_pca_by_cath_level(X_pca, merged_df, 'cath_T', 'Topology (C.A.T)', pca, axes[1, 0])
    plot_pca_by_cath_level(X_pca, merged_df, 'cath_H', 'Homology (C.A.T.H)', pca, axes[1, 1])
    
    plt.tight_layout()
    plt.savefig('pca_all_cath_levels.png', dpi=300, bbox_inches='tight')
    plt.show()
    return pca, X_pca, loadings_df

#run analysis
X_scaled, feature_cols, merged_df = prepare_features_with_cath(features_df, cath_df)
pca, X_pca, loadings = pca_analysis_all_levels(X_scaled, feature_cols, merged_df)

#save results
merged_df['PC1'] = X_pca[:, 0]
merged_df['PC2'] = X_pca[:, 1]
merged_df.to_csv('features_with_pca_all_cath.csv', index=False)
print(f"\n✓ Saved results with {len(merged_df)} proteins")