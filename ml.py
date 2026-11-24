import h5py
from sklearn.utils import shuffle
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, pairwise_distances
from scipy.stats import spearmanr
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from rdkit.Chem import MACCSkeys, RDKFingerprint
from sklearn.decomposition import PCA
import joblib
import pandas as pd 
from sklearn.model_selection import KFold

def mol_to_fp(smiles, fp_type, radius=2, n_bits=512):
    mol = Chem.MolFromSmiles(smiles)
    if fp_type == 'morgan':
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=radius,
            nBits=n_bits
        )
    elif fp_type == 'maccs':
        fp = MACCSkeys.GenMACCSKeys(mol)
    elif fp_type == 'rdkit':
        fp = RDKFingerprint(
            mol,
            fpSize=2048)
    arr = np.zeros((n_bits,), dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr

def calculate_max_similarity(X_general, X_target_train):
    X_general_bool = X_general.astype(bool)
    X_target_bool = X_target_train.astype(bool)
    similarity_matrix = 1- pairwise_distances(X_general_bool,X_target_bool,metric='jaccard')
    max_sims = similarity_matrix.max(axis=1)
    return max_sims

def canonicalize(s):
    s = s.replace('R1','1*').replace('R2','2*')
    return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)

def select_general_by_similarity(X_general, max_sims, threshold):
    if threshold <=0:
        return np.ones(len(X_general),dtype=bool)
    elif threshold >=1:
        return np.zeros(len(X_general),dtype=bool)
    else:
        return max_sims >= threshold

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

df = pd.read_csv('data/protac-fine4_ml.csv')
dff=df[df['logdc50'].notna()]
dfff = dff.groupby('Compound ID', as_index=False).agg({
    'Smiles': 'first',
    'logdc50': 'mean',
})

df_case = pd.read_csv('data/btk_baseline_new.csv') # change to your own target data
ddf_case = df_case[df_case['logdc50']!=6]
with open('quantitative_prediction.csv','w') as f:
    f.write('model,fp,case_choice,threshold,seed,r2,mae,rmse,spearman\n')
seeds = [0,10,42,66,88,100,666,2025,2026,666666]
fp_types = ['morgan','maccs','rdkit'] 
models = ['svr','gb','rf','xgboost'] 
thresholds = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1]
case_choices = [df_case, ddf_case]
for fp_type in fp_types:
    for model_type in models:
        for case in case_choices:
            for threshold in thresholds:
                for seed in seeds:
                    print(fp_type, model_type, threshold, seed)
                    setup_seed(seed)
                    X_general = np.stack(dfff.Smiles.apply(canonicalize).apply(mol_to_fp,fp_type=fp_type).values)
                    y_general = dfff.logdc50
                    X_target = np.stack(case.smiles.apply(canonicalize).apply(mol_to_fp,fp_type=fp_type).values)
                    y_target = case.logdc50.values

                    kf = KFold(n_splits=10,shuffle=True,random_state=seed)
                    oof_all = np.zeros(len(X_target))
                    for fold,(train_index, test_index) in enumerate(kf.split(X_target),1):
                        print(f'===Fold {fold}===')
                        X_train_target = np.array([X_target[i] for i in train_index], dtype=np.float32)
                        y_train_target = np.array([y_target[i] for i in train_index], dtype=np.float32)
                        X_test_target = np.array([X_target[i] for i in test_index], dtype=np.float32)
                        y_test_target = np.array([y_target[i] for i in test_index], dtype=np.float32)
                        max_sims = calculate_max_similarity(X_general, X_train_target)
                        selected_mask = select_general_by_similarity(X_general, max_sims, threshold)
                        X_general_selected = X_general[selected_mask]
                        y_general_selected = y_general[selected_mask]
                        X_train_all = np.vstack([X_train_target, X_general_selected])
                        y_train_all = np.concatenate([y_train_target, y_general_selected])
                        if model_type=='svr':
                            model = SVR(C=4, epsilon=0.03, gamma=0.01)
                        elif model_type=='gb':
                            model = GradientBoostingRegressor(learning_rate=0.2, n_estimators=1100,
                                             random_state=seed)
                        elif model_type=='rf':
                            model = RandomForestRegressor(n_estimators=500, random_state=seed)
                        elif model_type=='xgboost':
                            model = XGBRegressor(learning_rate=0.05, n_estimators=800, max_depth=7,gamma=0.1, 
                                                colsample_bytree=0.8, subsample=0.8, random_state=seed)
                        model.fit(X_train_all, y_train_all)
                        pred = model.predict(X_test_target)
                        oof_all[test_index] = pred

                    r2 = r2_score(y_target, oof_all)
                    mae = mean_absolute_error(y_target, oof_all)
                    rmse = np.sqrt(mean_squared_error(y_target, oof_all))
                    spearman = spearmanr(y_target, oof_all).correlation
                    print('R2: %.3f'%(r2))
                    print('MAE: %.3f'%(mae))
                    print('RMSE: %.3f'%(rmse)) 
                    print('Spearman: %.3f'%(spearman))

                    with open('quantitative_prediction.csv','a') as f:
                        f.write(f'{model_type},{fp_type},{len(case)},{threshold},{seed},{r2},{mae},{rmse},{spearman}\n')