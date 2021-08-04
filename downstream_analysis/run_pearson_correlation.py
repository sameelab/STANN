"""
   ___                  _                
  / _/______ ____  ____(_)__ _______     
 / _/ __/ _ `/ _ \/ __/ (_-</ __/ _ \    
/_//_/  \_,_/_//_/\__/_/___/\__/\___/    
  ___ _____(_)__ ___ ____  / /_(_)       
 / _ `/ __/ (_-</ _ `/ _ \/ __/ /        
 \_, /_/ /_/___/\_,_/_//_/\__/_/         
/___/
Samee Lab @ Baylor College Of Medicine
francisco.grisanticanozo@bcm.edu
Date: 06/2021
"""

import scipy
import pandas as pd

# Load Data
df = pd.read_csv("../outputs/STANN_predictions.csv")

n_cell_type = len(df.prediction.unique())
uni_cell_type = df.prediction.unique()
pcc = np.zeros(shape=(n_cell_type,n_cell_type),dtype=float)
pcc_p = np.zeros(shape=(n_cell_type,n_cell_type),dtype=float)


for j in range(n_cell_type):
    for i in range(n_cell_type):
        if i<j:     
            try:
                img1 = pd.read_csv("../outputs/results_ks/" + uni_cell_type[i]+ "_ks.csv")
                img2 = pd.read_csv("../outputs/results_ks/" + uni_cell_type[j]+ "_ks.csv")
                mask1 = img1 > np.quantile(img1.values, 0.5)
                mask2 = img2 > np.quantile(img2.values, 0.5)
                mask = ((mask1) & (mask2))
                v1 = img1.values[mask]
                v2 = img2.values[mask]
                
                temp_cor = round(scipy.stats.pearsonr(v1,v2)[0], 2)
                temp_p = round(scipy.stats.pearsonr(v1,v2)[1], 5)
                
                pcc[i,j] = temp_cor
                pcc_p[i,j] = temp_p
                
                pcc[j,i] = temp_cor
                pcc_p[j,i] = temp_p
                
            except:
                pcc[i,j] = "NaN"
                pcc_p[i,j] = "NaN"
        else:
            pcc[i,j] = 0
            pcc_p[i,j] = 0

df_pcc = pd.DataFrame(pcc, columns=uni_cell_type, index=uni_cell_type)
df_pcc.to_csv("Pearson correlation coefficient.csv")
df_pcc_p = pd.DataFrame(pcc_p, columns=uni_cell_type, index=uni_cell_type)
df_pcc_p.to_csv("Pearson correlation coefficient p value.csv")