# utils/featurize.py  — minimal, monocycle-only, no torch/pyg

from dataclasses import dataclass
from rdkit import Chem
import utils.chemistry.polygnn_kit as pk

# -------------------------
# Atom/Bond fingerprint API
# -------------------------
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise ValueError(f"input {x} not in allowable set {allowable_set}")
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# You can pare this element list further if you like
element_names = [
    "C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As","Al","I","B","V","K","Tl",
    "Yb","Sb","Sn","Ag","Pd","Co","Se","Ti","Zn","H","Li","Ge","Cu","Au","Ni","Cd","In","Mn",
    "Zr","Cr","Pt","Hg","Pb","Unknown",
]

@dataclass
class BondConfig:
    bond_type: bool
    conjugation: bool
    ring: bool
    stereo: bool = False
    bond_dir: bool = False

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []
        if self.bond_type:
            self.n_features += 4
            self.feat_names += ["Single","Double","Triple","Aromatic"]
        if self.conjugation:
            self.n_features += 1
            self.feat_names += ["Conjugation"]
        if self.ring:
            self.n_features += 1
            self.feat_names += ["inRing"]
        if self.stereo:
            self.n_features += 6
            self.feat_names += ["any","cis","e","none","trans","z"]
        if self.bond_dir:
            self.n_features += 7
            self.feat_names += [
                "begin_dash","begin_wedge","either_double","end_down_right",
                "end_up_right","none","unknown",
            ]

@dataclass
class AtomConfig:
    element_type: bool
    degree: bool
    implicit_valence: bool
    formal_charge: bool
    num_rad_e: bool
    hybridization: bool
    combo_hybrid: bool = False
    aromatic: bool = False
    chirality: bool = False

    def __post_init__(self):
        self.n_features = 0
        self.feat_names = []

        def add(names): self.feat_names += names; self.n_features += len(names)

        if self.element_type:     add(element_names)
        if self.degree:           add([f"degree{i}" for i in range(11)])
        if self.implicit_valence: add([f"implicitValence{i}" for i in range(7)])
        if self.formal_charge:    add(["formalCharge"])
        if self.num_rad_e:        add(["numRadElectons"])
        if self.hybridization:
            if self.combo_hybrid:
                add(["HybridizationType.SP","HybridizationType.SP2or3",
                     "HybridizationType.SP3D","HybridizationType.SP3D2"])
            else:
                add(["HybridizationType.SP","HybridizationType.SP2","HybridizationType.SP3",
                     "HybridizationType.SP3D","HybridizationType.SP3D2"])
        if self.aromatic:         add(["Aromatic"])
        if self.chirality:
            add(["Unspecified","Tetrahedral_CW","Tetrahedral_CCW","Other","Tetrahedral",
                 "Allene","Square_planar","Trigonal_bipyramidal","Octahedral"])

def bond_fp(bond, config: BondConfig):
    bt = bond.GetBondType()
    feats = []
    if config.bond_type:
        feats += [
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
        ]
    if config.conjugation: feats.append(bond.GetIsConjugated())
    if config.ring:        feats.append(bond.IsInRing())
    if config.stereo:
        st = bond.GetStereo()
        feats += [
            st == Chem.rdchem.BondStereo.STEREOANY,
            st == Chem.rdchem.BondStereo.STEREOCIS,
            st == Chem.rdchem.BondStereo.STEREOE,
            st == Chem.rdchem.BondStereo.STEREONONE,
            st == Chem.rdchem.BondStereo.STEREOTRANS,
            st == Chem.rdchem.BondStereo.STEREOZ,
        ]
    if config.bond_dir:
        feats += [
            bond.GetBondDir() == Chem.rdchem.BondDir.BEGINDASH,
            bond.GetBondDir() == Chem.rdchem.BondDir.BEGINWEDGE,
            bond.GetBondDir() == Chem.rdchem.BondDir.EITHERDOUBLE,
            bond.GetBondDir() == Chem.rdchem.BondDir.ENDDOWNRIGHT,
            bond.GetBondDir() == Chem.rdchem.BondDir.ENDUPRIGHT,
            bond.GetBondDir() == Chem.rdchem.BondDir.NONE,
            bond.GetBondDir() == Chem.rdchem.BondDir.UNKNOWN,
        ]
    return feats

def atom_fp(atom, cfg: AtomConfig):
    feats = []
    if cfg.element_type:
        feats += one_of_k_encoding_unk(atom.GetSymbol(), element_names)
    if cfg.degree:
        feats += one_of_k_encoding(atom.GetDegree(), list(range(11)))
    if cfg.implicit_valence:
        feats += one_of_k_encoding_unk(atom.GetImplicitValence(), list(range(7)))
    if cfg.formal_charge:
        feats += [atom.GetFormalCharge()]
    if cfg.num_rad_e:
        feats += [atom.GetNumRadicalElectrons()]
    if cfg.hybridization:
        h = atom.GetHybridization()
        if cfg.combo_hybrid and h in (Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3):
            h = "SP2/3"
            opts = [Chem.rdchem.HybridizationType.SP, "SP2/3",
                    Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]
        else:
            opts = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                    Chem.rdchem.HybridizationType.SP3D2]
        feats += one_of_k_encoding_unk(h, opts)
    if cfg.aromatic:
        feats += [atom.GetIsAromatic()]
    if cfg.chirality:
        tag = str(atom.GetChiralTag())
        feats += [
            tag == "CHI_UNSPECIFIED", tag == "CHI_TETRAHEDRAL_CW",
            tag == "CHI_TETRAHEDRAL_CCW", tag == "CHI_OTHER",
            tag == "CHI_TETRAHEDRAL", tag == "CHI_ALLENE",
            tag == "CHI_SQUAREPLANAR", tag == "CHI_TRIGONALBIPYRAMIDAL",
            tag == "CHI_OCTAHEDRAL",
        ]
    return feats

def atom_helper(molecule, idx, atom_config):
    return atom_fp(molecule.GetAtomWithIdx(idx), atom_config)

# -------------------------
# Monocycle helpers
# -------------------------
def get_valid_multiplications(smile: str, upper_bound: int):
    """Return minimal valid repeats (1..upper_bound) to allow cyclization."""
    if "[g]" in smile:
        return list(range(1, upper_bound + 1))
    valid = list(range(3, upper_bound + 1))
    mol = Chem.MolFromSmiles(smile)
    if   mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]~*~[#0]")): pass
    elif mol.HasSubstructMatch(Chem.MolFromSmarts("[#0]~*~*~[#0]")): valid.append(2)
    else: valid.extend([1, 2])
    return sorted(valid)

def multiply_and_cyclize(smile: str, n_repeat: int):
    """
    Build open parent with orig_idx (heavy atoms) and its cyclized monocycle.
    Returns: (mol_closed, mol_open)
    """
    polymer_class = pk.LadderPolymer if "[g]" in smile else pk.LinearPol
    lp = polymer_class(smile)
    mol_open = lp.multiply(n_repeat).mol
    for i, a in enumerate(mol_open.GetAtoms()):
        a.SetIntProp("orig_idx", i)
    pm = polymer_class(mol_open).PeriodicMol()
    mol_closed = pm.mol if hasattr(pm, "mol") else pm
    Chem.RemoveStereochemistry(mol_closed)
    Chem.RemoveStereochemistry(mol_open)
    return mol_closed, mol_open

def build_monocycle(smiles: str):
    valid = get_valid_multiplications(smiles, upper_bound=3)
    if not valid:
        raise ValueError("No valid multiplications for monocycle.")
    n_repeat = valid[0]
    mol_closed, mol_open = multiply_and_cyclize(smiles, n_repeat)

    if hasattr(mol_closed, "mol"):
        mol_closed = mol_closed.mol
    return mol_closed, mol_open
