import os
import numpy as np
from typing import List, Tuple, Optional

class DOS:
    """Parses the DOSCAR file generated from a VASP DOS calculation and provides methods to analyze the data.
    """
    def __init__(self):
        self.is_parsed = False
        self.is_spin_polarized = None
        self.fermi_energy = None
        self.energies = np.array([])
        self.energies_wrt_fermi = np.array([])
        self.total_dos_up = np.array([])
        self.total_dos_down = np.array([])
        self.partial_dos = None
    
    def parse_doscar(self):
        """Parses the DOSCAR obtained from the VASP DOS calculation.
        """
        if self.is_parsed:
            return
        assert os.path.exists("DOSCAR"), "DOSCAR is missing. DOS calculation is incomplete. Please check your calculation!"
        with open("DOSCAR", 'r') as f:
            lines = f.readlines()
        header = lines[5].split()
        self.fermi_energy = float(header[3])
        start = 6
        nedos = int(header[2])

        for line in lines[start:start+nedos]:
            values = np.array(list(map(float, line.split())))
            self.energies = np.append(self.energies, values[0])
            self.energies_wrt_fermi = self.energies - self.fermi_energy
            if values.size==3:
                self.is_spin_polarized = False
                self.total_dos_up = np.append(self.total_dos_up, values[1])
            elif values.size==5:
                self.is_spin_polarized = True
                self.total_dos_up = np.append(self.total_dos_up, values[1])
                self.total_dos_down = np.append(self.total_dos_down, values[2])

        if len(lines) > start+nedos:
            self.partial_dos = []
            n_atoms = int(lines[0].split()[0])
            partial_start = start+nedos+1
            for i in range(n_atoms):
                atom_partial_dos = []
                for line in lines[partial_start+i*nedos+i:partial_start+(i+1)*nedos+i]:
                    values = list(map(float, line.split()))
                    atom_partial_dos.append(values[1:])
                self.partial_dos.append(atom_partial_dos)
            self.partial_dos = np.array(self.partial_dos)
        self.is_parsed = True
    
    def get_band_gap(self) -> float:
        """Provides the band gap of the system from the parsed information.

        Returns:
            float: Band gap of the material.
        """
        self.parse_doscar()
        energies_below_fermi = self.energies[self.energies<self.fermi_energy]
        energies_above_fermi = self.energies[self.energies>self.fermi_energy]
        def gap_calc(dos_array):
            gap_start = None
            gap_end = None
            for energy, dos in zip(energies_below_fermi[::-1], dos_array[self.energies < self.fermi_energy][::-1]):
                if dos > 0:
                    gap_start = energy
                    break
            for energy, dos in zip(energies_above_fermi, dos_array[self.energies > self.fermi_energy]):
                if dos > 0:
                    gap_end = energy
                    break
            if gap_start is not None and gap_end is not None:
                return gap_end - gap_start
            else:
                return 0.0
        if self.is_spin_polarized:
            total_dos_up_and_down = self.total_dos_up + self.total_dos_down
        elif not self.is_spin_polarized:
            total_dos_up_and_down = self.total_dos_up
        return gap_calc(total_dos_up_and_down)
    
    def get_total_dos(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gives the total DOS of the system. For spin polarized calculations, it provides the total DOS of spin up and down channels. For non spin polarized calculations, the total DOS is provided in the spin up channel and the spin down channel is empty.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Total DOS of spin up and down channels.
        """
        self.parse_doscar()
        return self.total_dos_up, self.total_dos_down
    
    def get_orbital_projected_dos(self, orbital: str, dos_wrt_orb=None) -> Tuple[np.ndarray, np.ndarray]:
        """Gives the orbital projected DOS of the system. For spin polarized calculations, it provides the orbital projected DOS of spin up and down channels. For non spin polarized calculations, the orbital projected DOS is provided in the spin up channel and the spin down channel is empty.

        Args:
            orbital (str): Orbital on which the DOS is to be projected on, example: "d".
            dos_wrt_orb (_np.ndarray, optional): DOS with respect to different orbitals (s_up, s_down, px_up, px_down, py_up, py_down, ...). This is an internal parameter. Defaults to None.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Orbital projected DOS of spin up and down channels.
        """
        self.parse_doscar()
        if dos_wrt_orb is None:
            orbital_dos = sum(self.partial_dos)
        elif dos_wrt_orb is not None:
            orbital_dos = dos_wrt_orb
        orb_proj_dos_up = np.array([])
        orb_proj_dos_down = np.array([])
        if self.is_spin_polarized:
            if orbital=="s":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, row[0])
                    orb_proj_dos_down = np.append(orb_proj_dos_down, row[1])
            if orbital=="p":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[2:8:2]))
                    orb_proj_dos_down = np.append(orb_proj_dos_down, np.sum(row[3:8:2]))
            if orbital=="d":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[8:18:2]))
                    orb_proj_dos_down = np.append(orb_proj_dos_down, np.sum(row[9:18:2]))
            if orbital=="f":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[18:32:2]))
                    orb_proj_dos_down = np.append(orb_proj_dos_down, np.sum(row[19:32:2]))
        elif not self.is_spin_polarized:
            if orbital=="s":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, row[0])
            if orbital=="p":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[1:4:1]))
            if orbital=="d":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[4:9:1]))
            if orbital=="f":
                for row in orbital_dos:
                    orb_proj_dos_up = np.append(orb_proj_dos_up, np.sum(row[9:16:1]))
        return orb_proj_dos_up, orb_proj_dos_down
    
    def get_select_atoms_orbital_projected_dos(self, indices: List[int], orbital: str) -> Tuple[np.ndarray, np.ndarray]:
        """Gives the orbital projected DOS of select atoms in the system. For spin polarized calculations, it provides the orbital projected DOS of spin up and down channels. For non spin polarized calculations, the orbital projected DOS is provided in the spin up channel and the spin down channel is empty.
        
        Args:
            indices (List[int]): Indices of atoms for which the orbital projected DOS is to be calculated.
            orbital (str): Orbital on which the DOS is to be projected on, example: "d".

        Returns:
            Tuple[np.ndarray, np.ndarray]: Orbital projected DOS of select atoms of spin up and down channels.
        """
        self.parse_doscar()
        dos_wrt_orb = sum(self.partial_dos[i] for i in indices)
        atoms_orb_proj_dos_up, atoms_orb_proj_dos_down = self.get_orbital_projected_dos(orbital, dos_wrt_orb=dos_wrt_orb)
        return atoms_orb_proj_dos_up, atoms_orb_proj_dos_down
    
    def get_band_center(self, dos_up: np.ndarray, dos_down: Optional[np.ndarray]=None, energy_range: Optional[List[float]]=None) -> float:
        """Provides the band center of the DOS.

        Args:
            dos_up (np.ndarray): Spin up channel of DOS for which the band center is to be calculated.
            dos_down (Optional[np.ndarray], optional): Spin down channel of DOS for which the band center is to be calculated. Defaults to None.
            energy_range (Optional[List[float]], optional): Energy range of the DOS for which the band center is to be calculated. If None, the band center is calculated over the full energy range. Defaults to None.

        Returns:
            float: Band center of thr DOS.
        """
        self.parse_doscar()
        try:
            if energy_range is not None:
                energies, dos_up = self.get_dos_in_energy_range(dos_up, energy_range)
                _, dos_down = self.get_dos_in_energy_range(dos_down, energy_range)
            elif energy_range is None:
                energies = self.energies_wrt_fermi
            if dos_down is not None:
                dos_up_and_down = dos_up+dos_down
            else:
                dos_up_and_down = dos_up
            band_center = np.average(energies, weights=dos_up_and_down)
        except ZeroDivisionError:
            band_center = 0
        return band_center
    
    def get_dos_in_energy_range(self, dos: np.ndarray, energy_range: List[float]) -> Tuple[np.ndarray, np.ndarray]:
        """Filters the DOS in the specified energy range.

        Args:
            dos (np.ndarray): DOS which needs to be filtered to the custom energy range.
            energy_range (List[float]): Custom energy range to filter the DOS.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Filtered DOS.
        """
        self.parse_doscar()
        mask = (self.energies_wrt_fermi>=energy_range[0]) & (self.energies_wrt_fermi<=energy_range[1])
        return self.energies_wrt_fermi[mask], dos[mask]