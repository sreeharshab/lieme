import os
import re
from typing import List, Tuple, IO, Optional
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import Voronoi, _qhull
import ase
from ase import Atoms
from ase.io import read, Trajectory
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.calculators.calculator import Calculator
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from lieme.io import DOS, get_atoms_with_charges

"""
Li (li): Lithium
M (m): Metals
B (b): Bridging elements
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_formula_m_b(atoms: Atoms) -> Tuple[List[str], List[str]]:
    formula = str(atoms.symbols)
    elements = re.findall(r"([A-Z][a-z]?)\d*", formula)
    metals = [metal for metal in elements if metal not in ["O", "S", "C", "N", "Si", "P", "F", "Li"]]
    bridging_elements = [element for element in elements if element in ["O", "S", "C", "N", "Si", "P", "F"]]
    return formula, metals, bridging_elements

def relax(atoms: Atoms, calc: Calculator, fmax: float, trajectory: str=None) -> Atoms:
    atoms_copy = atoms.copy()
    atoms_copy.calc = calc
    ucf = UnitCellFilter(atoms_copy)
    opt = BFGS(ucf)
    if trajectory:
        traj = Trajectory(trajectory, "w")
        opt.attach(traj.write, 1, atoms_copy)
    opt.run(fmax=fmax)
    return atoms_copy

def get_relaxed_atoms_and_energy(dir_name: str="Energy_calculation", 
                                 atoms: Atoms=None, 
                                 calc: Calculator=None, 
                                 fmax: float=0.05
                                 ) -> Tuple[Atoms, float]:
    """Retrieves the Atoms object and the corresponding energy from the specified directory. 
        Either `OUTCAR` or `vasprun.xml` should be present in the directory.

    Args:
        dir_name (str, optional): Name of the directory. Defaults to "Energy_calculation".
        atoms (Atoms, optional): ASE Atoms object to be relaxed if `calc` is provided. Defaults to None.
        calc (Calculator, optional): ASE Calculator object used for atomic position relaxation. When None, 
                only feature extraction is performed from pre-existing DFT calculations. Defaults to None.
        fmax (float, optional): Maximum force criterion for relaxation. Defaults to 0.05 eV/Å.

    Returns:
        Tuple[Atoms, float]: Relaxed ASE Atoms object and the corresponding energy.
    """
    local_root = os.getcwd()
    trajectory = f"{dir_name}/opt.traj"
    try:
        os.chdir(dir_name)
        if os.path.exists("OUTCAR"):
            relaxed_atoms = read("OUTCAR@-1")
        elif os.path.exists("vasprun.xml"):
            relaxed_atoms = read("vasprun.xml")
        elif os.path.exists("opt.traj"):
            relaxed_atoms = read("opt.traj@-1")
        elif os.path.exists("POSCAR") and calc is not None:
            atoms = read("POSCAR")
            relaxed_atoms = relax(atoms=atoms, calc=calc, fmax=fmax, trajectory="opt.traj")
        else:
            raise IOError(f"Failed to read OUTCAR/vasprun.xml/POSCAR from `{dir_name}` at `{os.getcwd()}`.")
        energy = relaxed_atoms.get_potential_energy()
        return relaxed_atoms, energy
    except FileNotFoundError:
        if atoms is None:
            assert atoms is not None, "Atoms object must be provided during "
            f"initialization when it is not available at {dir_name}."
        assert calc is not None, "ASE Calculator must be provided when "
        f"Atoms object is not available at {dir_name}."
        if os.path.exists(str(trajectory)):
            relaxed_atoms = Trajectory(trajectory, "r")[-1]
            energy = relaxed_atoms.get_potential_energy()
        else:
            os.makedirs(trajectory.rsplit("/",1)[0], exist_ok=True)
            relaxed_atoms = relax(atoms, calc=calc, fmax=fmax, trajectory=trajectory)
            energy = relaxed_atoms.get_potential_energy()
        return relaxed_atoms, energy
    finally:
        os.chdir(local_root)

class GetFeatures:
    def __init__(self, 
                 material: str, 
                 atoms: Atoms=None,
                 calc: Calculator=None, 
                 fmax: float=0.05,
                 addnl_folder_paths: Optional[List[str]]=None,
                 ):
        """Initializes the GetFeatures class to extract features for a material.

        Args:
            material (str): Name of the material. It should be the same as the name of the directory 
                where all DFT calculations are stored.
            atoms (Atoms, optional): ASE Atoms object to be relaxed if `calc` is provided. Defaults to None.
            calc (Calculator, optional): ASE Calculator object used for atomic position relaxation. When None, 
                only feature extraction is performed from pre-existing DFT calculations. Defaults to None.
            fmax (float, optional): Maximum force criterion for relaxation. Defaults to 0.05 eV/Å.
            addnl_folder_paths (Optional[List[str]], optional): Additional folder paths other than the root 
                where the material's calculations can be found. Defaults to None.
        """
        self._root = os.getcwd()
        self.material = material
        logged = False
        if atoms is not None:
            self.formula, self.metals, self.bridging_elements = get_formula_m_b(atoms)
            logging.info(f"Material: {self.material}, Formula: {self.formula}")
            logged = True
        if atoms is not None and len(atoms)<10:
            logging.warning(f"The provided Atoms object for {material} has less than 10 atoms. "
                            "Repeating it to avoid errors during feature extraction.")
            repeat = int(np.ceil(10/len(atoms)))
            atoms = atoms*(repeat,1,1)
        self.atoms = atoms
        self.calc = calc
        self.fmax = fmax
        found = False
        if addnl_folder_paths is not None:
            addnl_folder_paths = [self._root] + addnl_folder_paths
            for base_path in addnl_folder_paths:
                target_dir = os.path.join(base_path, self.material)
                if os.path.exists(target_dir):
                    os.chdir(target_dir)
                    found = True
                    break
        if not found:
            target_dir = os.path.join(self._root, self.material)
            os.makedirs(target_dir, exist_ok=True)
            os.chdir(target_dir)
        try:
            self.relaxed_atoms = get_relaxed_atoms_and_energy(atoms=atoms, calc=calc, fmax=fmax)[0]
        except AssertionError:   # If you pass atoms, but do not pass calc and there is no calculation folder!
            self.relaxed_atoms = atoms
        if not logged and self.relaxed_atoms is not None:
            self.formula, self.metals, self.bridging_elements = get_formula_m_b(self.relaxed_atoms)
            logging.info(f"Material: {self.material}, Formula: {self.formula}")
        elif self.relaxed_atoms is None:
            raise ValueError("Relaxed Atoms object could not be obtained. Either "
            "`Energy_calculation` directory is missing or calc is not provided.")
        self.structure = AseAtomsAdaptor.get_structure(self.relaxed_atoms)
        self.data = None
    
    def get_voronoi_polyhedra_and_voronoi(self, structure: Structure, index: int, vnn: VoronoiNN) -> Tuple[dict, Voronoi]:
        """Adaptation of the VoronoiNN method from pymatgen. This is an internal method.
        """
        center = structure[index]
        targets = structure.elements if vnn.targets is None else vnn.targets
        corners = [[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]]
        d_corners = [np.linalg.norm(structure.lattice.get_cartesian_coords(c)) for c in corners]
        max_cutoff = max(d_corners) + 0.01
        while True:
            try:
                neighbors = structure.get_sites_in_sphere(center.coords, vnn.cutoff)
                neighbors = [ngbr[0] for ngbr in sorted(neighbors, key=lambda s: s[1])]
                qvoronoi_input = [site.coords for site in neighbors]
                voro = Voronoi(qvoronoi_input)
                cell_info = vnn._extract_cell_info(0, neighbors, targets, voro, vnn.compute_adj_neighbors)
                break
            except RuntimeError as exc:
                if vnn.cutoff >= max_cutoff:
                    if exc.args and "vertex" in exc.args[0]:
                        raise
                    raise RuntimeError("Error in Voronoi neighbor finding; max cutoff exceeded")
                vnn.cutoff = min(vnn.cutoff * 2, max_cutoff + 0.001)
        return cell_info, voro 
    
    def get_max_void_radius(self) -> float:
        """Provides the maximum void radius in the given structure.

        Returns:
            float: Maximum void radius of the structure.
        """
        structure = self.structure
        def distance(coord1, coord2):
            coord1 = np.array(coord1)
            coord2 = np.array(coord2)
            return np.linalg.norm(coord2 - coord1)
        vnn = VoronoiNN()
        radii = []
        for index,_ in enumerate(structure):
            voronoi_polyhedra, voronoi = self.get_voronoi_polyhedra_and_voronoi(structure, index, vnn)
            for poly_info in voronoi_polyhedra.values():
                vertex_indices = poly_info["verts"]
                vertices = voronoi.vertices[vertex_indices]
                for vertex in vertices:
                    radii.append(round(distance(vertex, structure[index].coords),3))
        return max(radii)
    
    def get_li_m_b_distances(self, atoms: Atoms=None, custom_cutoffs: Optional[dict]=None) -> List[float]:
        """Calculates the average distances between Li-M, Li-B, and M-B atoms.

        Args:
            atoms (Atoms): ASE Atoms object for which distances are to be calculated.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different atoms. Defaults to None.

        Returns:
            List[float]: The average distances [Li-M, Li-B, M-B].
        """
        if atoms is None:
            atoms = self.relaxed_atoms
        if custom_cutoffs:
            kwargs = custom_cutoffs
        else:
            kwargs = {"Mn":2, "Co":2, "Fe":2.5, "Nb":2, "C":1.7, "N":1.7}
        nat_cut = natural_cutoffs(atoms, **kwargs)
        nl = NeighborList(nat_cut, self_interaction=False, bothways=True)
        nl.update(atoms)
        li_m_distances, li_b_distances, m_b_distances = [], [], []
        li_indices = [atom.index for atom in atoms if atom.symbol=="Li"]
        if len(li_indices)!=0:
            try:
                for li_index in li_indices:
                    indices, offsets = nl.get_neighbors(li_index)
                    distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[li_index], axis=1)
                    li_m_distances.append(distances[[atoms[i].symbol in self.metals for i in indices]].mean())
                    li_b_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
            except RuntimeWarning:
                logging.warning(f"No neighbors found for Li atoms in {self.material}.\n"
                                "Try using `custom_cutoffs` to change the cutoffs for neighbor list computation.\n"
                                "Taking distances as 0...")
        else:            
            li_m_distances, li_b_distances = [0], [0]
        m_indices = [atom.index for atom in atoms if atom.symbol in self.metals]
        for m_index in m_indices:
            indices, offsets = nl.get_neighbors(m_index)
            distances = np.linalg.norm(atoms.positions[indices] + offsets @ atoms.cell - atoms.positions[m_index], axis=1)
            m_b_distances.append(distances[[atoms[i].symbol in self.bridging_elements for i in indices]].mean())
        return [round(np.mean(li_m_distances, axis=0),3), round(np.mean(li_b_distances, axis=0),3), round(np.mean(m_b_distances, axis=0),3)]
    
    def get_li_m_b_charges(self, dir_name: str="Bader_calculation") -> List[float]:
        """Computes the average charges on Li, M, and B atoms using Bader analysis from the specified directory. 
            Either `with_charges.traj` or `ACF.dat` should be present in the directory.

        Args:
            dir_name (str, optional): Name of the directory. Defaults to "Bader_calculation".

        Returns:
            List[float]: The average charges of [Li, M, B].
        """
        local_root = os.getcwd()
        try:
            os.chdir(dir_name)
            try:
                atoms = read("with_charges.traj")
            except FileNotFoundError:
                try:
                    atoms = get_atoms_with_charges()
                except IOError:
                    return [0,0,0]
            charges = atoms.get_initial_charges()
            li_charges = charges[[atom.symbol=="Li" for atom in atoms]]
            if len(li_charges)!=0:
                li_charge = round(np.mean(li_charges),3)
            else:
                li_charge = 0
            m_charges = charges[[atom.symbol in self.metals for atom in atoms]]
            m_charge = round(np.mean(m_charges),3)
            b_charges = charges[[atom.symbol in self.bridging_elements for atom in atoms]]
            b_charge = round(np.mean(b_charges),3)
            return [li_charge, m_charge, b_charge]
        except FileNotFoundError:
            return [0,0,0]
        finally:
            os.chdir(local_root)
    
    def get_dos_data(self, dir_name: str="Electronic_calculation", dos: Optional[CompleteDos]=None) -> List[float]:
        """Retrieves the band gap and band centers from the DOS data in the specified directory. 
            Either `DOSCAR` or `vasprun.xml` should be present in the directory.

        Args:
            dir_name (str, optional): Name of the directory. Defaults to "Electronic_calculation".
            dos (Optional[CompleteDos], optional): CompleteDos object from which band gap 
                and band centers are retrieved. This is an internal parameter. Defaults to None.

        Returns:
            List[float]: The band gap and band centers.
        """
        metals = self.metals
        bridging_elements = self.bridging_elements
        def get_band_centers(dos: DOS, dos_up: np.ndarray, dos_down: np.ndarray) -> List[float]:
            band_center = dos.get_band_center(dos_up, dos_down)
            val_band_center = dos.get_band_center(dos_up=dos_up, 
                                                  dos_down=dos_down, 
                                                  energy_range=[dos.energies_wrt_fermi[0],0])
            cond_band_center = dos.get_band_center(dos_up=dos_up, 
                                                   dos_down=dos_down, 
                                                   energy_range=[0,dos.energies_wrt_fermi[-1]])
            return [band_center, val_band_center, cond_band_center]
        if dos is not None:
            try:
                band_gap = dos.get_gap()
                energies = dos.energies - dos.efermi
                mask_val = energies < 0
                mask_cond = energies >= 0
                try:
                    tdos = dos.densities[Spin.up]+dos.densities[Spin.down]
                except KeyError:
                    tdos = dos.densities[Spin.up]
                band_centers = [np.average(energies, weights=tdos), 
                                np.average(energies[mask_val], weights=tdos[mask_val]), 
                                np.average(energies[mask_cond], weights=tdos[mask_cond])]
                p_band_centers = [dos.get_band_center(band=OrbitalType.p), 
                                  dos.get_band_center(band=OrbitalType.p, 
                                                      erange=[min(energies), 0]), 
                                  dos.get_band_center(band=OrbitalType.p, 
                                                      erange=[0, max(energies)])]
                d_band_centers = [dos.get_band_center(band=OrbitalType.d), 
                                  dos.get_band_center(band=OrbitalType.d, 
                                                      erange=[min(energies), 0]), 
                                  dos.get_band_center(band=OrbitalType.d, 
                                                      erange=[0, max(energies)])]
                metal_p_band_centers = [dos.get_band_center(band=OrbitalType.p, 
                                                            elements=metals), 
                                        dos.get_band_center(band=OrbitalType.p, 
                                                            elements=metals, 
                                                            erange=[min(energies), 0]), 
                                        dos.get_band_center(band=OrbitalType.p, 
                                                            elements=metals, 
                                                            erange=[0, max(energies)])]
                metal_d_band_centers = [dos.get_band_center(band=OrbitalType.d, 
                                                            elements=metals), 
                                        dos.get_band_center(band=OrbitalType.d, 
                                                            elements=metals, 
                                                            erange=[min(energies), 0]), 
                                        dos.get_band_center(band=OrbitalType.d, 
                                                            elements=metals, 
                                                            erange=[0, max(energies)])]
                brid_p_band_centers = [dos.get_band_center(band=OrbitalType.p, 
                                                           elements=bridging_elements), 
                                       dos.get_band_center(band=OrbitalType.p, 
                                                           elements=bridging_elements, 
                                                           erange=[min(energies), 0]), 
                                       dos.get_band_center(band=OrbitalType.p, 
                                                           elements=bridging_elements, 
                                                           erange=[0, max(energies)])]
                return ([band_gap] + band_centers + p_band_centers + d_band_centers + 
                        metal_p_band_centers + metal_d_band_centers + brid_p_band_centers)
            except Exception:
                return [0]*19   
        local_root = os.getcwd()
        try:
            os.chdir(dir_name)
            if os.path.exists("DOSCAR"):
                dos = DOS()
                band_gap = dos.get_band_gap()
                temp_dos_up, temp_dos_down = dos.get_total_dos()
                band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("p")
                p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("d")
                d_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.relaxed_atoms if atom.symbol in metals]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                metal_p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"d")
                metal_d_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.relaxed_atoms if atom.symbol in bridging_elements]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                brid_p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                return ([band_gap] + band_centers + p_band_centers + d_band_centers + 
                        metal_p_band_centers + metal_d_band_centers + brid_p_band_centers)
            elif os.path.exists("vasprun.xml"):
                vasprun = Vasprun("vasprun.xml")
                dos = vasprun.complete_dos
                return self.get_dos_data(dir_name=dir_name, dos=dos)
            else:
                return [0]*19
        except FileNotFoundError:
            return [0]*19
        finally:
            os.chdir(local_root)
    
    def get_intercalation_values(self, 
                                 atoms: float, 
                                 energy: float, 
                                 volume: float, 
                                 atoms_with_li: Atoms, 
                                 energy_with_li: float, 
                                 mu_li: float
                                 ) -> List[float]:
        """Internal method to compute Li intercalation prerequisites.
        """
        volume_with_li = atoms_with_li.get_volume()
        n_li = sum(1 for atom in atoms_with_li if atom.symbol=="Li") - sum(1 for atom in atoms if atom.symbol=="Li")
        if n_li==0:
            logging.warning("Number of intercalated Li atoms is zero. Taking intercalation values as 0...")
            return [0]*10
        li_energy = round((energy_with_li-energy-n_li*mu_li)/(n_li),3)
        volume_change = (volume_with_li - volume)/volume
        li_m_b_distances = self.get_li_m_b_distances(atoms=atoms_with_li)
        li_m_b_charges = self.get_li_m_b_charges(dir_name="bader")
        b_val_cond_band_centers = self.get_dos_data(dir_name="dos")[17:19]
        return [li_energy, volume_change]+li_m_b_distances+li_m_b_charges+b_val_cond_band_centers
    
    def get_intercalation_data(self,
                               li_m_ratios: List[float]=None,
                               mu_li: float=-2.076286119,
                               custom_n_m: Optional[dict]=None,
                               sampling_size: int=30,
                               seed: int=10,
                               li_atom_cutoff: float=1.7,
                               li_li_cutoff: float=1.0,
                               fhandle: Optional[IO]=None
                               ) -> List[float]:
        """Calculates the Li intercalation related properties for the material.

        Args:
            li_m_ratios (List[float], optional): List of Li/M ratios for which Li intercalation 
                features are to be calculated. Defaults to [0.25].
            mu_li (float, optional): The chemical potential of Li used to calculate the 
                Li intercalation energies. Defaults to -2.076286119 eV/atom.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present 
                in a material. Defaults to {"Li7NbS2": 8}.
            sampling_size (int, optional): Number of random intercalated structures to sample. Defaults to 30.
            seed (int, optional): Random seed for sampling intercalated structures. Defaults to 10.
            li_atom_cutoff (float, optional): Li-atom cutoff distance in intercalated structures. Defaults to 1.7 Å.
            li_li_cutoff (float, optional): Li-Li cutoff distance in intercalated structures. Defaults to 1.0 Å.
            fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.
        
        Returns:
            List[float]: Li intercalation features.
        """
        local_root = os.getcwd()
        atoms = self.relaxed_atoms
        energy = atoms.get_potential_energy()
        volume = atoms.get_volume()
        if li_m_ratios is None:
            li_m_ratios = [0.25]
        if custom_n_m is None:
            custom_n_m = {"Li7NbS2": 8}
        try:
            n_m = custom_n_m[self.material]
        except (KeyError, TypeError):
            n_m = sum(1 for atom in atoms if atom.symbol in self.metals)
        if fhandle:
            fhandle.write(f"Material: {self.material}, Formula: {self.formula}:\n")
            str_format = "{:^15} {:^28} {:^15} {:^15} {:^15} {:^23} {:^23} {:^23}\n"
            fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Average Li-M Distance", 
                                            "Average Li-B Distance", "Average M-B Distance", "Charge on Li", 
                                            "Charge on M", "Charge on B"))
        data = {}
        self.mless = {}  # mless: minimum Li energy sites
        try:
            nlifolders = [entry.name for entry in os.scandir("Intercalation") if entry.is_dir()]
            if self.calc is not None:
                assert len(nlifolders)==len(li_m_ratios), "Intercalation folders are not complete."
            os.chdir("Intercalation")
            for nlifolder in nlifolders:
                match = re.match(r"(\d+)_Li", nlifolder)
                if match:
                    n_li = int(match.group(1))
                else:
                    n_li = 0
                if not any(lower<=n_li/n_m<=upper for lower, upper in [(ratio-0.1, ratio+0.1) for ratio in li_m_ratios]):
                    continue
                fhandle.write(f"\tNumber of Li: {n_li}\n") if fhandle else None
                os.chdir(nlifolder)
                oswalk = [i for i in os.walk(".")]
                sites = sorted(oswalk[0][1])
                (li_energies, volume_changes, li_m_distances, li_b_distances, m_b_distances, 
                 li_charges, m_charges, b_charges, b_val_band_centers, b_cond_band_centers) = [[] for _ in range(10)]
                if self.calc is not None:
                    assert len(sites)==sampling_size, "Intercalation folders are not complete."
                for site in sites:
                    os.chdir(f"{site}")
                    try:
                        atoms_with_li, energy_with_li = get_relaxed_atoms_and_energy(dir_name="geo_opt")
                    except (FileNotFoundError, OSError):
                        os.chdir("../")
                        continue
                    lists = [li_energies, volume_changes, li_m_distances, li_b_distances, m_b_distances, 
                             li_charges, m_charges, b_charges, b_val_band_centers, b_cond_band_centers]
                    values = self.get_intercalation_values(atoms, energy, volume, atoms_with_li, energy_with_li, mu_li)
                    fhandle.write(str_format.format(site, values[0], values[2], values[3], values[4], values[5], 
                                                    values[6], values[7])) if fhandle else None
                    for lst, val in zip(lists, values):
                        lst.append(val)
                    os.chdir("../")
                mlei = li_energies.index(min(li_energies))  # mlei: minimum Li energy index
                if (li_charges[mlei]==0 and m_charges[mlei]==0 and b_charges[mlei]==0):
                    logging.warning(f"bader does not exist/is not completed at `{os.getcwd()}/{sites[mlei]}`. Taking charges as 0...")
                if (b_val_band_centers[mlei]==0 and b_cond_band_centers[mlei]==0):
                    logging.warning(f"dos does not exist/is not completed at `{os.getcwd()}/{sites[mlei]}`. Taking band centers as 0...")
                data[round(n_li/n_m,2)] = [li_energies[mlei], volume_changes[mlei], li_m_distances[mlei], 
                                           li_b_distances[mlei], m_b_distances[mlei], li_charges[mlei], 
                                           m_charges[mlei], b_charges[mlei], b_val_band_centers[mlei], 
                                           b_cond_band_centers[mlei]]
                for ratio in li_m_ratios:
                    if ratio-0.1<=n_li/n_m<=ratio+0.1:
                        data[ratio] = data[round(n_li/n_m,2)]
                        self.mless[ratio] = sites[mlei]
                os.chdir("../")
            os.chdir("../")
            fhandle.write("\n") if fhandle else None
            for ratio in li_m_ratios:
                if ratio not in data.keys():
                    logging.warning(f"Intercalation data does not exist for {ratio} Li/M at `{os.getcwd()}`. Taking values as 0...")
                    data[ratio] = [0]*10
            return sum([data[ratio] for ratio in li_m_ratios], [])
        except (FileNotFoundError, AssertionError):
            assert self.calc is not None, "ASE Calculator object must be provided when `Intercalation` directory is not complete."
            data = {}
            for ratio in li_m_ratios:
                intercalate = Intercalation(material=self.material, 
                                            atoms=atoms, 
                                            calc=self.calc, 
                                            fmax=self.fmax, 
                                            li_m_ratio=ratio, 
                                            custom_n_m=custom_n_m, 
                                            sampling_size=sampling_size, 
                                            seed=seed,
                                            li_atom_cutoff=li_atom_cutoff,
                                            li_li_cutoff=li_li_cutoff
                                            )
                atoms_with_li, energy_with_li = intercalate.get_best_intercalated_structure()
                if atoms_with_li is None:
                    data[ratio] = [0]*10
                    continue
                data[ratio] = self.get_intercalation_values(atoms, energy, volume, atoms_with_li, energy_with_li, mu_li)
            return sum([data[ratio] for ratio in li_m_ratios], [])
        finally:
            os.chdir(local_root)
    
    def get_data(self, 
                 custom_cutoffs: Optional[dict]=None,
                 li_m_ratios: List[float]=None,
                 mu_li: float=-2.076286119,
                 custom_n_m: Optional[dict]=None,
                 sampling_size: int=30,
                 seed: int=10,
                 li_atom_cutoff: float=1.7,
                 li_li_cutoff: float=1.0,
                 fhandle: Optional[IO]=None,
                 ) -> List[float]:
        """Returns the extracted features as a list.

        Args:
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for 
                different elements. Defaults to None.
            li_m_ratios (List[float], optional): List of Li/M ratios for which Li intercalation 
                features are to be calculated. Defaults to [0.25].
            mu_li (float, optional): The chemical potential of Li used to calculate the 
                Li intercalation energies. Defaults to -2.076286119 eV/atom.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present in 
                a material. Defaults to None.
            sampling_size (int, optional): Number of random intercalated structures to sample. Defaults to 30.
            seed (int, optional): Random seed for sampling intercalated structures. Defaults to 10.
            li_atom_cutoff (float, optional): Li-atom cutoff distance in intercalated structures. Defaults to 1.7 Å.
            li_li_cutoff (float, optional): Li-Li cutoff distance in intercalated structures. Defaults to 1.0 Å.
            fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.

        Returns:
            List[float]: Extracted features.
        """
        self.lattice_parameters = list(self.relaxed_atoms.cell.cellpar()[0:3]/self.relaxed_atoms.get_volume())
        self.max_void_radius = self.get_max_void_radius()
        self.distances = self.get_li_m_b_distances(atoms=self.relaxed_atoms, custom_cutoffs=custom_cutoffs)
        self.charges = self.get_li_m_b_charges()
        if self.charges==[0,0,0]:
            logging.warning(f"`Bader_calculation` does not exist/is not completed at `{os.getcwd()}`. Taking charges as 0...")
        self.dos_data = self.get_dos_data()
        if self.dos_data==[0]*19:
            logging.warning(f"`Electronic_calculation` does not exist/is not completed at `{os.getcwd()}`. "
                            "Taking band gap and band centers as 0...")
        self.intercalation_data = self.get_intercalation_data(li_m_ratios=li_m_ratios,
                                                              mu_li=mu_li,
                                                              custom_n_m=custom_n_m,
                                                              sampling_size=sampling_size,
                                                              seed=seed,
                                                              li_atom_cutoff=li_atom_cutoff,
                                                              li_li_cutoff=li_li_cutoff,
                                                              fhandle=fhandle,
                                                              )
        if self.intercalation_data==[0]*10*len(li_m_ratios):
            logging.warning(f"`Intercalation` does not exist/is not completed at `{os.getcwd()}`. " 
                            "Taking Li intercalation features as 0...")
        self.data = ([self.material, self.formula, self.structure] 
                     + self.lattice_parameters + [self.max_void_radius] 
                     + self.distances + self.charges + self.dos_data 
                     + self.intercalation_data)
    
    def return_to_root(self):
        """Returns to the root directory.
        """
        os.chdir(self._root)

def get_material_features(materials: List[str], 
                          atoms_list: Optional[List[Atoms]]=None, 
                          tag: Optional[str]=None, 
                          fhandle: Optional[IO]=None, 
                          addnl_folder_paths: Optional[List[str]]=None,
                          custom_cutoffs: Optional[dict]=None,
                          calc: Optional[Calculator]=None,
                          fmax: float=0.05,
                          li_m_ratios: List[float]=[0.25],
                          mu_li: float=-2.076286119,
                          custom_n_m: Optional[dict]=None,
                          sampling_size: int=30,
                          seed: int=10,
                          li_atom_cutoff: float=1.7,
                          li_li_cutoff: float=1.0
                          ) -> DataFrame:
    """Extracts features for a list of materials using the GetFeatures class.

    Args:
        materials (List[str]): List of material names. Each material should have a 
            directory with the same name where all DFT calculations are stored.
        atoms_list (Optional[List[Atoms]], optional): List of ASE Atoms objects corresponding
            to the materials. If not provided, the Atoms objects will be read from the
            calculation directories. Defaults to None.
        tag (Optional[str], optional): Features are saved in a file named `material_features_{tag}.pkl` 
            if `tag` is provided, otherwise in `material_features.pkl`. Defaults to None.
        fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.
        addnl_folder_paths (Optional[List[str]], optional): Additional folder paths other than 
            the root where the material's calculations can be found. Defaults to None.
        custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for 
            different elements. Defaults to None.
        calc (Optional[Calculator], optional): ASE Calculator object to be used for
            relaxation if calculation files are not found. Defaults to None.
        fmax (float, optional): Maximum force criterion for relaxation. Defaults to 0.05 eV/Å.
        li_m_ratios (List[float], optional): List of Li/M ratios for which Li intercalation 
            features are to be calculated. Defaults to [0.25].
        mu_li (float, optional): The chemical potential of Li used to calculate the Li 
            intercalation energies. Defaults to -2.076286119 eV/atom.
        custom_n_m (Optional[dict], optional): Custom number of metal atoms present in a material. Defaults to None.
        sampling_size (int, optional): Number of random intercalated structures to sample. Defaults to 30.
        seed (int, optional): Random seed for sampling intercalated structures. Defaults to 10.
        li_atom_cutoff (float, optional): Li-atom cutoff distance in intercalated structures. Defaults to 1.7 Å.
        li_li_cutoff (float, optional): Li-Li cutoff distance in intercalated structures. Defaults to 1.0 Å.
        
    Returns:
        DataFrame: Features for all materials.
    """
    base_cols = [
        "material", "formula", "structure", "Lattice Parameter a", "Lattice Parameter b", "Lattice Parameter c", 
        "Maximum Void Radius", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", 
        "Charge on Li", "Charge on M", "Charge on B", "Band Gap", "Band Center", "Valence Band Center", 
        "Conduction Band Center", "p Band Center", "Valence p Band Center", "Conduction p Band Center", 
        "d Band Center", "Valence d Band Center", "Conduction d Band Center", "M p Band Center", 
        "M Valence p Band Center", "M Conduction p Band Center", "M d Band Center", "M Valence d Band Center", 
        "M Conduction d Band Center", "B p Band Center", "B Valence p Band Center", "B Conduction p Band Center"
    ]
    intercalation_cols = []
    for ratio in li_m_ratios:
        suffix = f"@ {ratio:.2f} Li/M"
        intercalation_cols += [
            f"Li Intercalation Energy {suffix}",
            f"Volume Change {suffix}",
            f"Average Li-M Distance {suffix}",
            f"Average Li-B Distance {suffix}",
            f"Average M-B Distance {suffix}",
            f"Charge on Li {suffix}",
            f"Charge on M {suffix}",
            f"Charge on B {suffix}",
            f"B Valence p Band Center {suffix}",
            f"B Conduction p Band Center {suffix}",
        ]
    df = pd.DataFrame(columns=base_cols+intercalation_cols)
    atoms_list = atoms_list if atoms_list is not None else [None]*len(materials)
    for material, atoms in zip(materials, atoms_list):
        features = GetFeatures(material=material, 
                               atoms=atoms,
                               calc=calc,
                               fmax=fmax,
                               addnl_folder_paths=addnl_folder_paths)
        features.get_data(
            custom_cutoffs=custom_cutoffs,
            li_m_ratios=li_m_ratios,
            mu_li=mu_li,
            custom_n_m=custom_n_m,
            sampling_size=sampling_size,
            seed=seed,
            li_atom_cutoff=li_atom_cutoff,
            li_li_cutoff=li_li_cutoff,
            fhandle=fhandle
        )
        next_index = len(df)
        df.loc[next_index] = features.data
        features.return_to_root()
    df = StrToComposition().featurize_dataframe(df, "formula")
    ep_feat = ElementProperty.from_preset(preset_name="magpie")
    df = ep_feat.featurize_dataframe(df, col_id="composition") 
    df_feat = DensityFeatures()
    df = df_feat.featurize_dataframe(df, col_id="structure")
    mpe_feat = MaximumPackingEfficiency()
    df = mpe_feat.featurize_dataframe(df, col_id="structure") 
    file_name = f"material_features_{tag}.pkl" if tag else "material_features.pkl"
    df.to_pickle(file_name)
    return df

class Intercalation:
    def __init__(self, 
                 material: str, 
                 atoms: Atoms, 
                 calc: Calculator, 
                 fmax: float,
                 li_m_ratio: float,
                 custom_n_m: Optional[dict]=None,
                 sampling_size: int=30,
                 seed: int=10,
                 li_atom_cutoff: float=1.7,
                 li_li_cutoff: float=1.0
                 ):
        self.material = material
        self.atoms = atoms
        self.calc = calc
        self.fmax = fmax
        self.li_m_ratio = li_m_ratio
        self.custom_n_m = custom_n_m
        self.sampling_size = sampling_size
        self.seed = seed
        self.li_atom_cutoff = li_atom_cutoff
        self.li_li_cutoff = li_li_cutoff

    def generate_intercalated_structures(self) -> List[Structure]:
        li_m_ratio = self.li_m_ratio
        sampling_size = self.sampling_size
        if self.seed is not None:    
            np.random.seed(self.seed)
        li_structures = []
        atoms = self.atoms
        structure = AseAtomsAdaptor.get_structure(atoms)
        custom_n_m = self.custom_n_m
        if custom_n_m is None:
            custom_n_m = {"Li7NbS2": 8}
        try:
            n_m = custom_n_m[self.material]
        except (KeyError, TypeError):
            _, metals, _ = get_formula_m_b(atoms)
            n_m = sum(1 for atom in atoms if atom.symbol in metals)
        coords = np.array([site.coords for site in structure.sites])
        try:
            voro = Voronoi(coords)
        except _qhull.QhullError:
            voro = Voronoi(coords, qhull_options="QJ")
        frac_sites = []
        for v in voro.vertices:
            f = structure.lattice.get_fractional_coords(v)
            if np.all(f>=0) and np.all(f<=1):
                frac_sites.append(f)
        def is_valid(frac_site):
            c = structure.lattice.get_cartesian_coords(frac_site)
            dists = [np.linalg.norm(c - site.coords) for site in structure.sites]
            return all(dist>self.li_atom_cutoff for dist in dists)
        def filter_valid_sites(frac_sites):
            valid_sites = []
            for frac_site in frac_sites:
                if not is_valid(frac_site):
                    continue
                c = structure.lattice.get_cartesian_coords(frac_site)
                if not valid_sites or all(np.linalg.norm(c - structure.lattice.get_cartesian_coords(vs))>self.li_li_cutoff
                                          for vs in valid_sites):
                    valid_sites.append(frac_site)
            return valid_sites
        valid_sites = filter_valid_sites(frac_sites)
        n_li = max(1, int(round(li_m_ratio*n_m)))
        count = 0
        matcher = StructureMatcher()
        for _ in range(sampling_size):
            if n_li>len(valid_sites):
                logging.info(f"Not enough interstitial sites for Li/M={li_m_ratio}, returning None!")
                return
            idxs = np.random.choice(len(valid_sites), n_li, replace=False)
            li_structure = structure.copy()
            for idx in idxs:
                li_structure.append("Li", valid_sites[idx])
            if count>0:
                is_different = not any(matcher.fit(li_structure, existing_structure) 
                                   for existing_structure in li_structures)
                if is_different:
                    li_structures.append(li_structure)
                    count+=1
            else:
                li_structures.append(li_structure)
                count+=1
        logging.info(f"Generated {count} unique intercalated structures for Li/M={li_m_ratio}...")
        return li_structures
    
    def get_best_intercalated_structure(self) -> Structure:
        li_m_ratio = self.li_m_ratio
        dir = "Intercalation"
        traj_path = f"{dir}/sampling_{li_m_ratio}.traj"
        os.makedirs(dir, exist_ok=True)
        if os.path.exists(f"{dir}/sampling_{li_m_ratio}.traj"):
            try:
                traj_read = Trajectory(traj_path, "r")
            except ase.io.ulm.InvalidULMFileError:
                traj_read = None
            mode = "a"
        else:
            mode = "w"
        traj = Trajectory(f"{dir}/sampling_{li_m_ratio}.traj", mode)
        structures_with_li = self.generate_intercalated_structures()
        atoms_list_with_li = []
        energies_with_li = []
        if not structures_with_li:
            return None, None
        for index, structure_with_li in enumerate(structures_with_li):
            if index < len(traj):
                atoms_list_with_li.append(traj_read[index])
                energies_with_li.append(traj_read[index].get_potential_energy())
                continue
            # Manual atoms construction since using adaptor is causing issues during relaxation!
            positions = [site.coords for site in structure_with_li.sites]
            symbols = [site.specie.symbol for site in structure_with_li.sites]
            cell = structure_with_li.lattice.matrix
            atoms_with_li = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
            n_li = sum(1 for atom in atoms_with_li if atom.symbol=="Li") - sum(1 for atom in self.atoms if atom.symbol=="Li")
            relax_dir = f"Intercalation/{n_li}_Li/{index}/geo_opt"
            atoms_with_li, energy_with_li = get_relaxed_atoms_and_energy( 
                dir_name=relax_dir,
                atoms=atoms_with_li,
                calc=self.calc,
                fmax=self.fmax
                )
            atoms_list_with_li.append(atoms_with_li)
            energies_with_li.append(energy_with_li)
            traj.write(atoms_with_li)
        traj.close()
        traj_read.close() if 'traj_read' in locals() else None
        min_idx = np.argmin(energies_with_li)
        return atoms_list_with_li[min_idx], energies_with_li[min_idx]