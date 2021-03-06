import numpy as np
from rdkit import Chem
import torch
import re
from transformers import BertModel, BertTokenizer
import esm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PROTBERT_LOADED = False
ESM1B_LOADED = False


def one_hot_encode(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]

    encoded = [
        int(bool_value)
        for bool_value in list(map(lambda s: x == s, allowable_set))
    ]

    return encoded


def truncate(sequence: str, max_len: int = 1000):

    if len(sequence) >= max_len:
        return sequence[:max_len]
    else:
        return sequence


def get_atom_features(atom, use_chirality=True, hydrogens_implicit=True):
    """
    Takes an RDKit atom object as input and gives a 1d-numpy array of atom features as output.
    """

    # define list of permitted atoms

    permitted_list_of_atoms = [
        'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe',
        'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd',
        'Co', 'Se', 'Ti', 'Zn', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn',
        'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
    ]

    if hydrogens_implicit == False:
        permitted_list_of_atoms = ['H'] + permitted_list_of_atoms

    # compute atom features

    atom_type_enc = one_hot_encode(str(atom.GetSymbol()),
                                   permitted_list_of_atoms)

    n_heavy_neighbors_enc = one_hot_encode(int(atom.GetDegree()),
                                           [0, 1, 2, 3, 4, "MoreThanFour"])

    formal_charge_enc = one_hot_encode(int(atom.GetFormalCharge()),
                                       [-3, -2, -1, 0, 1, 2, 3, "Extreme"])

    hybridisation_type_enc = one_hot_encode(
        str(atom.GetHybridization()),
        ["S", "SP", "SP2", "SP3", "SP3D", "SP3D2", "OTHER"])

    is_in_a_ring_enc = [int(atom.IsInRing())]

    is_aromatic_enc = [int(atom.GetIsAromatic())]

    atomic_mass_scaled = [float((atom.GetMass() - 10.812) / 116.092)]

    vdw_radius_scaled = [
        float(
            (Chem.GetPeriodicTable().GetRvdw(atom.GetAtomicNum()) - 1.5) / 0.6)
    ]

    covalent_radius_scaled = [
        float(
            (Chem.GetPeriodicTable().GetRcovalent(atom.GetAtomicNum()) - 0.64) /
            0.76)
    ]

    atom_feature_vector = atom_type_enc + n_heavy_neighbors_enc + formal_charge_enc + hybridisation_type_enc + is_in_a_ring_enc + is_aromatic_enc + atomic_mass_scaled + vdw_radius_scaled + covalent_radius_scaled

    if use_chirality == True:
        chirality_type_enc = one_hot_encode(str(atom.GetChiralTag()), [
            "CHI_UNSPECIFIED", "CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW",
            "CHI_OTHER"
        ])
        atom_feature_vector += chirality_type_enc

    if hydrogens_implicit == True:
        n_hydrogens_enc = one_hot_encode(int(atom.GetTotalNumHs()),
                                         [0, 1, 2, 3, 4, "MoreThanFour"])
        atom_feature_vector += n_hydrogens_enc

    return np.array(atom_feature_vector)


def get_bond_features(bond, use_stereochemistry=True):
    """
    Takes an RDKit bond object as input and gives a 1d-numpy array of bond features as output.
    """

    permitted_list_of_bond_types = [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ]

    bond_type_enc = one_hot_encode(bond.GetBondType(),
                                   permitted_list_of_bond_types)

    bond_is_conj_enc = [int(bond.GetIsConjugated())]

    bond_is_in_ring_enc = [int(bond.IsInRing())]

    bond_feature_vector = bond_type_enc + bond_is_conj_enc + bond_is_in_ring_enc

    if use_stereochemistry == True:
        stereo_type_enc = one_hot_encode(
            str(bond.GetStereo()),
            ["STEREOZ", "STEREOE", "STEREOANY", "STEREONONE"])
        bond_feature_vector += stereo_type_enc

    return np.array(bond_feature_vector)


def get_protbert_embedding(sequence: str, max_len: int = 1000) -> list:

    global PROTBERT_LOADED

    if not PROTBERT_LOADED:
        protbert_tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert',
                                                           do_lower_case=False)
        protbert_model = BertModel.from_pretrained("Rostlab/prot_bert")

    protbert_model.to(device)
    protbert_model.eval()

    sequences = [sequence]
    sequences = [
        re.sub(r"[UZOB]", "X", truncate(seq, max_len)) for seq in sequences
    ]
    sequences = [" ".join(x) for x in sequences]

    ids = protbert_tokenizer.batch_encode_plus(sequences,
                                               add_special_tokens=True,
                                               padding='longest')
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding = protbert_model(input_ids=input_ids,
                                   attention_mask=attention_mask)[0]

    embedding = embedding.cpu().numpy()

    seq_len = (attention_mask[0] == 1).sum()
    seq_emd = embedding[0][1:seq_len - 1]
    features = seq_emd.tolist()

    PROTBERT_LOADED = True

    return features


def get_esm1b_embedding(sequence: str, max_len: int = 1000) -> list:

    global ESM1B_LOADED

    if not ESM1B_LOADED:
        esm1b_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        esm1b_batch_converter = alphabet.get_batch_converter()

    esm1b_model.eval()
    esm1b_model.to(device)

    sequences = [sequence]
    data = [('', truncate(seq, max_len)) for seq in sequences]
    batch_labels, batch_strs, batch_tokens = esm1b_batch_converter(data)

    # Extract per-residue representations
    with torch.no_grad():
        results = esm1b_model(batch_tokens.to(device),
                              repr_layers=[33],
                              return_contacts=True)
    token_representations = results["representations"][33]
    token_representations = token_representations.cpu().numpy()

    _, seq = data[0]
    features = token_representations[0, 1:len(seq) + 1].tolist()

    ESM1B_LOADED = True

    return features