import os
import re
import numpy as np
import pandas as pd
from ase import Atoms
from ase.io import read, write

def get_atoms_with_charges() -> Atoms:
    """Converts ACF.dat from Bader charge analysis to ASE Atoms object with charges.

    Returns:
        Atoms: ASE Atoms object with initial charges set based on Bader charge analysis.
    """
    valence_electrons = {}
    try:
        f = open("POTCAR", "r")
    except:
        raise IOError(f"Failed to read POTCAR from {os.getcwd()}.")
    lines = f.readlines()
    search_str = lines[0].split()[0]
    for i,line in enumerate(lines):
        if search_str in line and not any(excluded in line for excluded in ["TITEL", "LPAW", "radial sets"]):
            match = re.search(fr"{search_str}\s+([A-Z][a-z]?)", line)
            next_line = lines[i+1]
            nelect = int(float(next_line.split()[0]))
        if match:
            element = match.group(1)
        valence_electrons[element] = nelect
    try:
        atoms = read("OUTCAR")
    except:
        try:
            atoms = read("vasprun.xml")
        except:
            raise IOError(f"Failed to read OUTCAR/vasprun.xml from {os.getcwd()}.")
    latoms = len(atoms)
    try:
        df = pd.read_table(
            "ACF.dat",
            delim_whitespace=True,
            header=0,
            skiprows=[1, latoms + 2, latoms + 3, latoms + 4, latoms + 5],
        )
    except:
        raise IOError(f"Failed to read ACF.dat from {os.getcwd()}.")
    charges = df["CHARGE"].to_numpy()
    ocharges = np.array([])
    for atom in atoms:
        ocharges = np.append(ocharges, valence_electrons[atom.symbol])
    ocharges = np.array([int(i) for i in ocharges])
    dcharges = -charges + ocharges
    atoms.set_initial_charges(np.round(dcharges, 2))
    write("with_charges.traj", atoms)
    return atoms