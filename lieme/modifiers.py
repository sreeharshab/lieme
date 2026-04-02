import random
import numpy as np
from typing import Dict, List
from ase import Atoms, Atom
from gg.modifiers.modifiers import ParentModifier

class ResolvePartialOccupancies(ParentModifier):
    """
    This is a modifier which can be used in Basin Hopping (refer to 
    https://graph-gcbh.readthedocs.io/en/latest/examples.html) to obtain the 
    best site occupancies (resolving partial occupancies).
    atoms (current structure) -> modified_atoms (new structure) with site information from 
    parent_atoms (parent structure).
    Parent structure: Structure with all sites occupied. For example, Li9La3Ta2O12
    Current structure: Random structure with target composition. For example, Li6BaLa2Ta2O12
    Modified structure: New random structure with target composition.
    """
    def __init__(self, 
                 parent_atoms: Atoms, 
                 weight: float, 
                 site_occupancy: Dict[str, List], 
                 n_swaps: int = 2, 
                 position_tolerance: float = 0.5
                 ):
        """
        Args:
            parent_atoms (Atoms): Parent structure with site information.
            weight (float): Weight for this modifier.
            site_occupancy (Dict[str, List]): Dict mapping parent symbols to modifiable indices. 
                For example, {"Ge": [0, 1, 2, ..., 19],
                              "Li": [20, 21, ..., 29]}
            n_swaps (int): Number of site swaps to perform. Defaults to 2.
            position_tolerance (float): Distance tolerance for mapping current sites to parent sites.
        """
        super().__init__(weight=weight)
        self.parent_atoms = parent_atoms
        self.site_occupancy = site_occupancy
        self.n_swaps = n_swaps
        self.position_tolerance = position_tolerance
    
    def _map_to_parent_sites(self, atoms: Atoms) -> Dict[int, int]:
        """
        Maps each atom in atoms to its corresponding parent site index.
        
        Returns: 
            Dict[int, int]: current_site_index -> parent_site_index map.
        """
        parent_atoms = self.parent_atoms
        current_to_parent = {}
        parent_frac_positions = parent_atoms.get_scaled_positions()
        current_frac_positions = atoms.get_scaled_positions()
        parent_cell = parent_atoms.get_cell()
        pbc = parent_atoms.get_pbc()
        for current_idx, current_frac_position in enumerate(current_frac_positions):
            min_dist = float("inf")
            best_parent_idx = None
            for parent_idx, parent_frac_position in enumerate(parent_frac_positions):
                frac_diff = current_frac_position - parent_frac_position
                if any(pbc):
                    frac_diff = frac_diff - np.round(frac_diff)
                cart_diff = np.dot(frac_diff, parent_cell.array) # Reference of parent cell, 
                                                                 # can also consider current cell
                dist = np.linalg.norm(cart_diff)
                if dist < min_dist:
                    min_dist = dist
                    best_parent_idx = parent_idx
            if min_dist <= self.position_tolerance:
                current_to_parent[current_idx] = best_parent_idx
        return current_to_parent
    
    def get_modified_atoms(self, atoms: Atoms) -> Atoms:
        """
        Generate modified structure by swapping n_swaps site occupancies.
        
        Args:
            atom (Atoms): Current structure with target composition.
        
        Returns:
            Atoms: Modified structure with same composition but different site occupancies.
        """
        parent_atoms = self.parent_atoms
        modified_atoms = atoms.copy()
        current_to_parent = self._map_to_parent_sites(atoms)
        parent_to_current = {v: k for k, v in current_to_parent.items()}
        current_occupancy = {}
        for parent_symbol, indices in self.site_occupancy.items():
            for parent_idx in indices:
                current_occupancy[parent_idx] = "X"
        for current_idx, parent_idx in current_to_parent.items():
            if parent_idx in current_occupancy:
                current_occupancy[parent_idx] = modified_atoms[current_idx].symbol
        for _ in range(self.n_swaps):
            parent_symbol = random.choice(list(self.site_occupancy.keys()))
            indices = self.site_occupancy[parent_symbol]
            if len(indices) < 2:
                continue
            site1, site2 = random.sample(indices, 2)
            current_occupancy[site1], current_occupancy[site2] = \
                current_occupancy[site2], current_occupancy[site1]
        indices_to_remove = []
        for parent_idx, symbol in current_occupancy.items():
            if symbol == "X":
                if parent_idx in parent_to_current:
                    indices_to_remove.append(parent_to_current[parent_idx])
            else:
                if parent_idx in parent_to_current:
                    modified_atoms[parent_to_current[parent_idx]].symbol = symbol
                else:
                    position = parent_atoms[parent_idx].position
                    new_atom = Atom(symbol, position)
                    modified_atoms.append(new_atom)
        indices_to_remove.sort(reverse=True)
        del modified_atoms[indices_to_remove]
        return modified_atoms