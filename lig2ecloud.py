import h5py
import numpy as np
from src.chem import get_mol_center
from src.ecloud_utils.xtb_density import CDCalculator, interplot_ecloud
from src.ecloud_utils.grid import BuildGridCenters
from tqdm import tqdm
from rdkit import Chem
from transformers import T5TokenizerFast

def protocol(grid, resolution=0.5):
    '''
    Define the grid protocol, including grid size, resolution, and grid centers
        grid size is the number of grids in each dimension
        resolution is the distance between two grid points

    '''
    N = [grid, grid, grid]
    llc = (np.zeros(3) - float(grid * resolution / 2)) + resolution / 2  # lower left corner
    grids = BuildGridCenters(llc, N, resolution)    

    return {'grids':grids, 'N':N, 'llc':llc}

def get_ligecloud(mol, calculater, protocol):
    '''
    Input:
        mol: rdkit 3D mol
        calculater: xtb density calculater
        protocol: protocol for the grid, format: {'grids':(32, 32, 32, 3), 'N':[32, 32, 32]}
        add_noise: add noise to the ligand grid
    Output:
        lig_density: ligand electron density, shape: (32, 32, 32)
    '''
    stand_grid = protocol['grids']
    N = protocol['N']
    mol_center = get_mol_center(mol) 
    lig_grids = stand_grid + mol_center 
    lig_ecloud = calculater.calculate(mol)
    lig_density = interplot_ecloud(lig_ecloud, lig_grids.transpose(3, 0, 1, 2)).reshape(N)
    return lig_density

def single_process(mol):
    calculater = CDCalculator(xtb_command='xtb')
    try:
        ecloud = get_ligecloud(mol, calculater, protocol(grid=64))
        s = Chem.MolToSmiles(mol)
        # print(s)
        return [ecloud, s]
    except:
        return None

max_input_len = 200 # 256
tokenizer = T5TokenizerFast.from_pretrained("tokenizer")

mols = [m for m in Chem.SDMolSupplier('data/protac-fine4.sdf')]
with open('data/protac-fine4.csv') as f:
    data = [line.strip('\n').split(',') for line in f][1:]
df = {}
for inf in data:
    df[inf[1]] = [inf[0],inf[5],inf[6]] # id, smi, label

e = []
input = []
ids = []
cla = []

for m in tqdm(mols):
    name = m.GetProp('_Name')
    id,smi,label = df[name]
    pair = single_process(m)
    e.append(pair[0])

    tokened_src = tokenizer(
            smi,
            max_length=max_input_len,
            truncation=True,
            padding="max_length",
        )["input_ids"]

    input.append(tokened_src)
    ids.append(id)
    if label == 'TRUE':
        cla.append(1)
    elif label == 'FALSE':
        cla.append(0)
    else:
        raise ValueError('check label: ', label)

print('Data num: ', len(e))
with h5py.File('protac-fine4.h5','w') as f_:
    eclouds=f_.create_dataset("eclouds", (len(e),64,64,64), dtype='f')
    input_ids=f_.create_dataset("input_ids", (len(e),max_input_len), dtype='i')
    classes=f_.create_dataset("classes", (len(e),1), dtype='i')
    labels=f_.create_dataset("labels", (len(e),1), dtype='i')
    
    for i in range(len(e)):
        eclouds[i]=e[i]
        input_ids[i]=input[i]
        classes[i]=cla[i]
        labels[i]=ids[i]



