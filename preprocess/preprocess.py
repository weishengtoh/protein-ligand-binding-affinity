import numpy as np
from typing import Callable, Dict, Any, List
import torch
from rdkit import Chem
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch_geometric.data import Data
from dgllife.utils import smiles_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from utils.utils import get_atom_features, get_bond_features, get_protbert_embedding, get_esm1b_embedding
from model.basemodel import Models
import dgl


class PreprocessGen:

    @staticmethod
    def get_prepro_fn(model: Models,
                      smiles: str,
                      sequence: str,
                      label: str = None) -> Callable:
        global _SMILES
        global _SEQUENCE
        global _LABEL

        _SMILES = smiles
        _SEQUENCE = sequence
        _LABEL = label

        if model == Models.CNN_CNN:
            return PreprocessGen.smiles_label_seq_label
        elif model == Models.GCN_CNN:
            return PreprocessGen.smiles_graph_seq_label
        elif model == Models.CNN_TRANSFORMERS:
            return PreprocessGen.smiles_label_seq_label
        elif model == Models.GAT_TRANSFORMERS:
            return PreprocessGen.smiles_graph_seq_label
        elif model == Models.GAT_CNN:
            return PreprocessGen.smiles_graph_seq_label
        elif model == Models.GAT_TRANSFORMERS:
            return PreprocessGen.smiles_graph_seq_label
        elif model == Models.GCN_ESM1B:
            return PreprocessGen.smiles_graph_seq_esm1b
        elif model == Models.GCN_PROTBERT:
            return PreprocessGen.smiles_graph_seq_protbert
        elif model == Models.GINMASK_CNN:
            return PreprocessGen.smiles_gin_seq_label
        elif model == Models.GINCONTEXTPRED_CNN:
            return PreprocessGen.smiles_gin_seq_label
        elif model == Models.GINEDGEPRED_CNN:
            return PreprocessGen.smiles_gin_seq_label
        elif model == Models.GININFOMAX_CNN:
            return PreprocessGen.smiles_gin_seq_label
        elif model == Models.GINMASK_PROTBERT:
            return PreprocessGen.smiles_gin_seq_protbert
        elif model == Models.GINCONTEXTPRED_PROTBERT:
            return PreprocessGen.smiles_gin_seq_protbert
        elif model == Models.GINEDGEPRED_PROTBERT:
            return PreprocessGen.smiles_gin_seq_protbert

    @staticmethod
    def get_collate_fn(model) -> Callable:
        if model == Models.GINMASK_CNN:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GINCONTEXTPRED_CNN:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GINEDGEPRED_CNN:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GININFOMAX_CNN:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GINMASK_PROTBERT:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GINCONTEXTPRED_PROTBERT:
            return PreprocessGen.dgl_collate_fn
        elif model == Models.GINEDGEPRED_PROTBERT:
            return PreprocessGen.dgl_collate_fn
        else:
            return None

    @staticmethod
    def dgl_collate_fn(sample: List[Dict[str, Any]]):
        smiles = [x[_SMILES] for x in sample]
        smiles = dgl.batch(smiles)

        sequence = [x[_SEQUENCE] for x in sample]
        sequence = torch.stack(sequence)

        result = {_SMILES: smiles, _SEQUENCE: sequence}

        label = [
            x[_LABEL] for x in sample if not np.isnan(x.get(_LABEL, np.nan))
        ]
        if len(label) > 0:
            result.update({_LABEL: torch.stack(label)})

        return result

    @staticmethod
    def smiles_gin_seq_label(sample: Dict[str, Any]) -> Dict[str, torch.tensor]:

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        seq_voc = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
        ]
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        max_seq_len = 1000

        # Convert a SMILES into a bi-directed DGLGraph and featurize for it
        node_featurizer = PretrainAtomFeaturizer()
        edge_featurizer = PretrainBondFeaturizer()
        data = smiles_to_bigraph(smiles=smiles,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 add_self_loop=True)

        # construct sequence tensor
        seq_label_encoded = np.zeros(max_seq_len)
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_encoded[i] = seq_dict[ch]

        seq_tensor = torch.tensor(seq_label_encoded, dtype=torch.int)

        result = {_SMILES: data, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result

    @staticmethod
    def smiles_label_seq_label(
            sample: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """Preprocess data for DeepDTA based models"""

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        seq_voc = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
        ]
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        max_seq_len = 1000

        smiles_voc = [
            '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
            '8', '9', '=', '@', 'A', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P',
            'S', 'V', 'Z', '[', '\\', ']', 'c', 'e', 'i', 'l', 'n', 'o', 'r',
            's', '.'
        ]
        smiles_dict = {v: (i + 1) for i, v in enumerate(smiles_voc)}
        max_smiles_len = 100

        # construct sequence tensor
        seq_label_encoded = np.zeros(max_seq_len)
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_encoded[i] = seq_dict[ch]

        seq_tensor = torch.tensor(seq_label_encoded, dtype=torch.int)

        # construct smiles tensor
        smiles_label_encoded = np.zeros(max_smiles_len)
        for i, ch in enumerate(smiles[:max_smiles_len]):
            smiles_label_encoded[i] = smiles_dict[ch]

        smiles_tensor = torch.tensor(smiles_label_encoded, dtype=torch.int)

        result = {_SMILES: smiles_tensor, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result

    @staticmethod
    def smiles_graph_seq_label(
            sample: Dict[str, Any]) -> Dict[str, torch.tensor]:
        """Preprocess data for GraphDTA based models"""

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        seq_voc = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
            'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X'
        ]
        seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
        max_seq_len = 1000

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(
            unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(
            get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):

            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct sequence tensor
        seq_label_encoded = np.zeros(max_seq_len)
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_encoded[i] = seq_dict[ch]

        seq_tensor = torch.tensor(seq_label_encoded, dtype=torch.int)

        # construct Pytorch Geometric data object and append to data list
        data = Data(x=X, edge_index=E, edge_attr=EF, y=None)

        result = {_SMILES: data, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result

    @staticmethod
    def smiles_gin_seq_protbert(
            sample: Dict[str, Any]) -> Dict[str, torch.tensor]:

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        # Convert a SMILES into a bi-directed DGLGraph and featurize for it
        node_featurizer = PretrainAtomFeaturizer()
        edge_featurizer = PretrainBondFeaturizer()
        data = smiles_to_bigraph(smiles=smiles,
                                 node_featurizer=node_featurizer,
                                 edge_featurizer=edge_featurizer,
                                 add_self_loop=True)

        # construct sequence tensor
        seq = get_protbert_embedding(seq)
        max_seq_len = 1000
        seq_label_padded = np.zeros((max_seq_len, len(seq[0])))
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_padded[i] = ch

        seq_tensor = torch.tensor(seq_label_padded, dtype=torch.float)

        result = {_SMILES: data, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result

    @staticmethod
    def smiles_graph_seq_protbert(
            sample: Dict[str, Any]) -> Dict[str, torch.tensor]:

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(
            unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(
            get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):

            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct sequence tensor
        seq = get_protbert_embedding(seq)
        max_seq_len = 1000
        seq_label_padded = np.zeros((max_seq_len, len(seq[0])))
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_padded[i] = ch

        seq_tensor = torch.tensor(seq_label_padded, dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data = Data(x=X, edge_index=E, edge_attr=EF, y=None)

        result = {_SMILES: data, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result

    @staticmethod
    def smiles_graph_seq_esm1b(
            sample: Dict[str, Any]) -> Dict[str, torch.tensor]:

        smiles = sample[_SMILES]
        seq = sample[_SEQUENCE]

        # convert SMILES to RDKit mol object
        mol = Chem.MolFromSmiles(smiles)

        # get feature dimensions
        n_nodes = mol.GetNumAtoms()
        n_edges = 2 * mol.GetNumBonds()
        unrelated_smiles = "O=O"
        unrelated_mol = Chem.MolFromSmiles(unrelated_smiles)
        n_node_features = len(get_atom_features(
            unrelated_mol.GetAtomWithIdx(0)))
        n_edge_features = len(
            get_bond_features(unrelated_mol.GetBondBetweenAtoms(0, 1)))

        # construct node feature matrix X of shape (n_nodes, n_node_features)
        X = np.zeros((n_nodes, n_node_features))

        for atom in mol.GetAtoms():
            X[atom.GetIdx(), :] = get_atom_features(atom)

        X = torch.tensor(X, dtype=torch.float)

        # construct edge index array E of shape (2, n_edges)
        (rows, cols) = np.nonzero(GetAdjacencyMatrix(mol))
        torch_rows = torch.from_numpy(rows.astype(np.int64)).to(torch.long)
        torch_cols = torch.from_numpy(cols.astype(np.int64)).to(torch.long)
        E = torch.stack([torch_rows, torch_cols], dim=0)

        # construct edge feature array EF of shape (n_edges, n_edge_features)
        EF = np.zeros((n_edges, n_edge_features))

        for (k, (i, j)) in enumerate(zip(rows, cols)):

            EF[k] = get_bond_features(mol.GetBondBetweenAtoms(int(i), int(j)))

        EF = torch.tensor(EF, dtype=torch.float)

        # construct sequence tensor
        seq = get_esm1b_embedding(seq)
        max_seq_len = 1000
        seq_label_padded = np.zeros((max_seq_len, len(seq[0])))
        for i, ch in enumerate(seq[:max_seq_len]):
            seq_label_padded[i] = ch

        seq_tensor = torch.tensor(seq_label_padded, dtype=torch.float)

        # construct Pytorch Geometric data object and append to data list
        data = Data(x=X, edge_index=E, edge_attr=EF, y=None)

        result = {_SMILES: data, _SEQUENCE: seq_tensor}

        # construct label tensor
        if _LABEL is not None:
            y_val = sample[_LABEL]
            y_tensor = torch.tensor(np.array(y_val), dtype=torch.float)

            result.update({_LABEL: y_tensor})

        return result
