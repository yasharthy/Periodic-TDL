from rdkit import Chem
import numpy as np

def n_to_subtract(atoms_removed_inds, atom_ind):
    return int(np.argwhere(np.sort(atoms_removed_inds + [atom_ind]) == atom_ind)[0][0])



class LinearPol(Chem.rdchem.Mol):
    """
    Linear Polymer class.

    Attributes:
        mol (Chem.rdchem.Mol): The RDKit molecule of the parent polymer.
        star_inds (list): List of star atom indices in the parent polymer. Sorted from smallest to largest index.
        connector_inds (list): List of connector atom indices in the parent polymer. Sorted from smallest to largest index.
        periodic_bond_type (Chem.rdchem.BondType): The bond type of the periodic bond in the parent polymer.
    """

    def __init__(self, mol):
        """
        Initializes the LinearPol object.

        Args:
            mol (str or Chem.rdchem.Mol): SMILES string or RDKit molecule of the parent polymer.
        """
        if isinstance(mol, str):
            self.SMILES = mol
            self.mol = Chem.MolFromSmiles(mol)
        else:
            self.mol = mol

        self.star_inds, self.connector_inds = get_star_inds(self.mol)
        self.periodic_bond_type = self.get_pbond_type()

    def get_pbond_type(self):
        """
        Returns the bond type of the periodic bond.

        Returns:
            Chem.rdchem.BondType: The bond type of the periodic bond.

        Raises:
            ValueError: If the bond types of the periodic bond are mismatching.
        """
        bond1_type = self.mol.GetBondBetweenAtoms(self.star_inds[0], self.connector_inds[0]).GetBondType()
        bond2_type = self.mol.GetBondBetweenAtoms(self.star_inds[1], self.connector_inds[1]).GetBondType()
        if bond1_type != bond2_type:
            raise ValueError("Invalid repeat unit. Periodic bond types are mismatching.")
        return bond1_type

    def PeriodicMol(self, repeat_unit_on_fail=False):
        """
        Creates a periodic representation of the linear polymer molecule.

        Args:
            repeat_unit_on_fail (bool, optional): Whether to return None or attempt to create a periodic molecule
                even if the periodization fails. Defaults to False.

        Returns:
            PeriodicMol or None: The periodic representation of the linear polymer molecule, or None if the
                periodization fails.
        """
        em = Chem.EditableMol(self.mol)
        try:
            em.AddBond(self.connector_inds[0], self.connector_inds[1], self.periodic_bond_type)
            em.RemoveAtom(self.star_inds[1])
            em.RemoveAtom(self.star_inds[0])
            return PeriodicMol(em.GetMol(), self.star_inds, self.connector_inds)
        except:
            if not repeat_unit_on_fail:
                return None
            else:
                em.RemoveAtom(self.star_inds[1])
                em.RemoveAtom(self.star_inds[0])
                return PeriodicMol(em.GetMol(), self.star_inds, self.connector_inds)

    def multiply(self, n):
        """
        Return a LinearPol which is n times repeated from itself
        """
        add_lp = LinearPol(self.mol)
        for _ in range(n - 1):
            add_lp = LinearPol(
                bind_frags(
                    self.mol,
                    [self.star_inds[1]],
                    add_lp.mol,
                    [add_lp.star_inds[0]],
                    [self.connector_inds[1]],
                    [add_lp.connector_inds[0]],
                    [self.periodic_bond_type],
                )
            )
        return add_lp


def bind_frags(
    m1,
    m1_tails,
    m2,
    m2_heads,
    m1_connectors,
    m2_connectors,
    bond_types,
):
    """
    Bind *mol* objects, m1 and m2, together with a bond of type 'bond_type' at connection points m1_tail and m2_head
    """

    combo_mol = Chem.rdmolops.CombineMols(m1, m2)
    em = Chem.EditableMol(combo_mol)
    removed_atoms = [np.inf]  # init with 'inf' so that n_to_subtract will be 0 to start
    for (m1_connector, m2_connector, m1_tail, m2_head, bond_type) in zip(
        m1_connectors, m2_connectors, m1_tails, m2_heads, bond_types
    ):
        em.AddBond(
            m1_connector - n_to_subtract(removed_atoms, m1_connector),
            m2_connector
            + m1.GetNumAtoms()
            - n_to_subtract(removed_atoms, m2_connector + m1.GetNumAtoms()),
            bond_type,
        )
        em.RemoveAtom(m1_tail - n_to_subtract(removed_atoms, m1_tail))
        removed_atoms.append(m1_tail)
        em.RemoveAtom(
            m2_head
            + m1.GetNumAtoms()
            - n_to_subtract(removed_atoms, m2_head + m1.GetNumAtoms())
        )
        removed_atoms.append(m2_head + m1.GetNumAtoms())

    new_mol = em.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
        return new_mol
    except:
        print(f"!!! Binding failed for {Chem.MolToSmiles(combo_mol)} !!!")
        return None


class PeriodicMol(Chem.rdchem.Mol):
    """
    Periodic representation of Linear Polymer class.
    """

    def __init__(self, mol, star_inds, connector_inds):
        self.mol = mol
        # self.connector_inds = []
        # for i in connector_inds:
        #     srt = np.sort(star_inds + [i])[::-1]  # sort in reverse order
        #     new_ind = i - int(np.argwhere(srt == i))
        #     self.connector_inds.append(new_ind)

    def HasSubstructMatch(self, match_mol):
        try:
            return self.mol.HasSubstructMatch(match_mol)
        except:
            self.GetSSSR()
            return self.mol.HasSubstructMatch(match_mol)

    def GetSubstructMatches(self, match_mol):
        try:
            return self.mol.GetSubstructMatches(match_mol)
        except:
            self.GetSSSR()
            return self.mol.GetSubstructMatches(match_mol)

    def GetSubstructMatch(self, match_mol):
        try:
            return self.mol.GetSubstructMatch(match_mol)
        except:
            self.GetSSSR()
            return self.mol.GetSubstructMatch(match_mol)

    def GetSSSR(self):
        Chem.GetSSSR(self.mol)

    def GetRingInfo(self):
        return self.mol.GetRingInfo()


def get_star_inds(mol):
    if type(mol) == str:
        mol = Chem.MolFromSmiles(mol)
    inds = tuple(mol.GetSubstructMatches(Chem.MolFromSmarts("[#0]~*")))
    inds = tuple(zip(*inds))
    if len(inds[0]) != 2:
        raise ValueError(
            "Invalid repeat unit. It is likely that at least one star contains more than one bond."
        )
    return list(inds[0]), list(inds[1])


def ladder_smiles(smiles, e_repl, t_repl, g_repl, d_repl):

    return (
        smiles.replace("[e]", e_repl)
        .replace("[t]", t_repl)
        .replace("[g]", g_repl)
        .replace("[d]", d_repl)
    )


class LadderPolymer(LinearPol):
    def __init__(self, mol) -> None:
        # e,t share periodic bond
        # d,g share perdioic bond
        # e,d share "side"
        # t,g share "side"
        self.isotope_e = "200"
        self.isotope_t = "201"
        self.isotope_d = "300"
        self.isotope_g = "301"
        if isinstance(mol, str):
            mol = ladder_smiles(
                mol,
                f"[{self.isotope_e}*]",
                f"[{self.isotope_t}*]",
                f"[{self.isotope_d}*]",
                f"[{self.isotope_g}*]",
            )
            mol = Chem.MolFromSmiles(mol)

        self.mol = mol
        e_ind, ec_ind = self.mol.GetSubstructMatch(
            Chem.MolFromSmarts(f"[{self.isotope_e}#0]~*")
        )
        t_ind, tc_ind = self.mol.GetSubstructMatch(
            Chem.MolFromSmarts(f"[{self.isotope_t}#0]~*")
        )
        d_ind, dc_ind = self.mol.GetSubstructMatch(
            Chem.MolFromSmarts(f"[{self.isotope_d}#0]~*")
        )
        g_ind, gc_ind = self.mol.GetSubstructMatch(
            Chem.MolFromSmarts(f"[{self.isotope_g}#0]~*")
        )
        star_to_connector = {
            e_ind: ec_ind,
            t_ind: tc_ind,
            d_ind: dc_ind,
            g_ind: gc_ind,
        }
        star_to_bound_star = {
            e_ind: t_ind,
            t_ind: e_ind,
            d_ind: g_ind,
            g_ind: d_ind,
        }
        star_to_side_star = {e_ind: d_ind, d_ind: e_ind, t_ind: g_ind, g_ind: t_ind}
        # A1, A2 share periodic bond
        # B1, B2 share perdioic bond
        # A1, B1 share "side"
        # A2, B2 share "side"
        self.starA1_ind = min([e_ind, t_ind, d_ind, g_ind])
        self.connectorA1_ind = star_to_connector[self.starA1_ind]
        self.starA2_ind = star_to_bound_star[self.starA1_ind]
        self.connectorA2_ind = star_to_connector[self.starA2_ind]
        self.starB1_ind = star_to_side_star[self.starA1_ind]
        self.connectorB1_ind = star_to_connector[self.starB1_ind]
        self.starB2_ind = star_to_bound_star[self.starB1_ind]
        self.connectorB2_ind = star_to_connector[self.starB2_ind]

        self.periodic_bond1_type = self.mol.GetBondBetweenAtoms(
            self.starA1_ind, self.connectorA1_ind
        ).GetBondType()
        self.periodic_bond2_type = self.mol.GetBondBetweenAtoms(
            self.starB1_ind, self.connectorB1_ind
        ).GetBondType()

    def PeriodicMol(self):
        em = Chem.EditableMol(self.mol)
        try:
            em.AddBond(
                self.connectorA1_ind, self.connectorA2_ind, self.periodic_bond1_type
            )
            em.AddBond(
                self.connectorB1_ind, self.connectorB2_ind, self.periodic_bond2_type
            )
            remove_inds = sorted(
                [self.starA1_ind, self.starA2_ind, self.starB1_ind, self.starB2_ind]
            )
            em.RemoveAtom(remove_inds[-1])
            em.RemoveAtom(remove_inds[-2])
            em.RemoveAtom(remove_inds[-3])
            em.RemoveAtom(remove_inds[-4])
            return em.GetMol()
        except:
            print("!!!Periodization of %s Failed!!!" % self.SMILES)
            return None

    def multiply(self, n):
        """
        Return a LinearPol which is n times repeated from itself
        """
        add_lp = LadderPolymer(self.mol)
        for _ in range(n - 1):
            add_lp = LadderPolymer(
                bind_frags(
                    m1=self.mol,
                    m1_tails=[self.starA2_ind, self.starB2_ind],
                    m2=add_lp.mol,
                    m2_heads=[add_lp.starA1_ind, add_lp.starB1_ind],
                    m1_connectors=[self.connectorA2_ind, self.connectorB2_ind],
                    m2_connectors=[add_lp.connectorA1_ind, add_lp.connectorB1_ind],
                    bond_types=[self.periodic_bond1_type, self.periodic_bond2_type],
                )
            )
        return add_lp
