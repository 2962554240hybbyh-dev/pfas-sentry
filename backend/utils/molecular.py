"""
分子解析和特征提取模块
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
import base64
import io


def parse_smiles(smiles):
    """解析SMILES字符串"""
    mol = Chem.MolFromSmiles(smiles)
    return mol


def validate_smiles(smiles):
    """验证SMILES是否有效"""
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None


def get_molecular_properties(mol):
    """获取分子性质"""
    if mol is None:
        return {}

    n_f = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 9)
    n_c = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
    n_o = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)
    n_s = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 16)

    return {
        'molecular_formula': Chem.rdMolDescriptors.CalcMolFormula(mol),
        'molecular_weight': round(Descriptors.MolWt(mol), 2),
        'logp': round(Descriptors.MolLogP(mol), 2),
        'tpsa': round(Descriptors.TPSA(mol), 2),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
        'heavy_atoms': mol.GetNumHeavyAtoms(),
        'fluorine_count': n_f,
        'carbon_count': n_c,
        'oxygen_count': n_o,
        'sulfur_count': n_s,
        'f_c_ratio': round(n_f / max(n_c, 1), 2),
        'f_heavy_ratio': round(n_f / max(mol.GetNumHeavyAtoms(), 1), 2),
    }


def generate_descriptors(smiles):
    """生成分子描述符向量"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    desc = []

    # RDKit描述符
    for name, func in Descriptors.descList:
        try:
            val = func(mol)
            desc.append(float(val) if val and not np.isinf(val) else 0.0)
        except:
            desc.append(0.0)

    # Morgan指纹 (256位)
    try:
        morgan = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=256))
        desc.extend(morgan)
    except:
        desc.extend([0] * 256)

    # MACCS指纹 (167位)
    try:
        maccs = list(MACCSkeys.GenMACCSKeys(mol))
        desc.extend(maccs)
    except:
        desc.extend([0] * 167)

    return np.array(desc).reshape(1, -1)


def get_mol_image_base64(smiles, size=(400, 300)):
    """获取分子结构图的base64编码"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    try:
        img = Draw.MolToImage(mol, size=size)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except:
        return None
