import numpy as np
from ase import Atoms

def repeat_to_n_atoms(
        atoms: Atoms,
        n_atoms: int
        ) -> Atoms:
    """Repeat the unit cell to create a supercell with at least n_atoms.
    The cell is repeated preferentially along the smallest dimension.
    
    Args:
        atoms (Atoms): ASE Atoms object.
        n_atoms (int): Minimum number of atoms in the supercell.
    
    Returns:
        Atoms: Supercell ASE Atoms object.
    """
    atoms = atoms.copy()
    current_n_atoms = len(atoms)
    cell = atoms.get_cell()
    a, b, c = cell.lengths()
    n_repeats = np.array([1, 1, 1])
    while current_n_atoms*np.prod(n_repeats)<n_atoms:
        supercell_lengths = n_repeats*np.array([a, b, c])
        min_idx = np.argmin(supercell_lengths)
        n_repeats[min_idx] += 1
    supercell = atoms.repeat(tuple(n_repeats))
    return supercell