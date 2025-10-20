##feature vectors: properties of the writhe matrices that are put into a vector which allows the machine to notice patterns more easily 
##introduced by paper: RT Journal Article, A1 Hou, Jie, DeepSF: deep convolutional neural network for mapping protein sequences to folds 2017,  Bioinformatics, 10.1093/bioinformatics/btx780
##and : https://doi.org/10.1146/annurev-statistics-031017-100045; Topological Data Analysis paper

## key idea: check for natural groupings (unsupervised learning)
# get feature vectors that show the important information of each writhe matrices, 
# use PCA/t-SNE to project 15D feature vectors into 2D/3D and create scatter plots 
# PC1 - weighted combination of all the features that will capture maximum variance
# PC2 - another weighted combination that captures second most variance 
# can make intrepretations of PC1 and PC2 as like "overall entanglement intensity" or " pattern complexity"
# this will be helpful in seeing how complex the data variation is so that choosing the ML architecture is easier
# to select an architecture I would need to know: data variance, data size, class imbalances, data quality...


import numpy as np
import pandas as pd
import os
from scipy import stats

def create_feature_vectors(gauss_folder, df):
    
    features_list = [] #stores proteins 
    missing_files = [] #for failed files 
    
    for idx, row in df.iterrows():
        file = f"{row['pdb_id']}-{row['chain_id']}.npz"  #construct file name as: PDB ID + chain ID eg: 101m-A.npz
        filepath = os.path.join(gauss_folder, file)
        
        if os.path.exists(filepath):
            data = np.load(filepath)
            matrix = data['matrix'] #extract the writhe matrix 
            n = matrix.shape[0]  #matrix size for number of rows/columns = protein length 
            
            #extract key features
            features = {
                'pdb_id': row['pdb_id'],
                'chain_id': row['chain_id'],
                'protein_id': row['protein_id'],
                'matrix_size': n,
                'residues': row['residue_count'],  #residues = amino acids in protein chain 
                
                #basic statistical features
                'writhe_mean': np.mean(matrix),
                'writhe_std': np.std(matrix),
                'writhe_min': np.min(matrix),
                'writhe_max': np.max(matrix),
                'writhe_range': np.ptp(matrix),   #peak to peak: max - min function
                
                #distribution features
                'writhe_skew': stats.skew(matrix.flatten()), #asymmetry of writhe value distribution
                'writhe_kurtosis': stats.kurtosis(matrix.flatten()), #measures how extreme outliers are ##flatten for 2D array to 1D array 
                

                'diagonal_mean': np.mean(np.diag(matrix)), #self writhe should be 0 
                'off_diag_mean': np.mean(matrix[np.triu_indices_from(matrix, k=1)]), #entanglement strength (from upper triangle only) 
                #k =0 diagonal 
                #k = 1 one step above diagonal 
                #k = 2 two steps above diagonal 

                'abs_mean': np.mean(np.abs(matrix)), #mean absolute value of matrix - how much entanglement in total 
                'entropy': stats.entropy(np.histogram(matrix.flatten(), bins=10)[0] + 1e-10), #entropy will tell us how many bins the values are clustered into; the lower the entropy the simplier the patternXZ
                
                'norm_mean': np.mean(matrix) / n if n > 0 else 0, #writhe PER residue so that its bigger protein having bigger writhe doesn't make a difference ee
                'complexity': np.mean(np.abs(matrix)) * np.std(matrix) #mean times the std for complexity of pattern
            }
            features_list.append(features)
        else:
            missing_files.append(file)
    
    #create DataFrame
    features_df = pd.DataFrame(features_list)
    return features_df

features_df = create_feature_vectors('Protein Gauss', df)

print("Protein's Feature Vectors:")
print(features_df.to_string(max_rows=10)