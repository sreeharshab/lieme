import os
import re
from typing import List, Tuple, IO, Optional
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial import Voronoi
from ase import Atoms
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from pymatgen.core import Structure
from pymatgen.electronic_structure.core import OrbitalType, Spin
from pymatgen.electronic_structure.dos import CompleteDos
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from lieme.io import DOS, get_atoms_with_charges

"""
Li (li): Lithium
M (m): Metals
B (b): Bridging elements
"""

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
root = os.getcwd()

class GetFeatures:
    def __init__(self, material: str, fhandle: Optional[IO]=None, addnl_folder_paths: Optional[List[str]]=None, use_methods_only: bool=False, custom_cutoffs: Optional[dict]=None, mu_li: float=-2.076286119, custom_n_m: Optional[dict]=None):
        """Initializes the GetFeatures class to extract features for a material.

        Args:
            material (str): Name of the material. It should be the same as the name of the directory where all DFT calculations are stored.
            fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.
            addnl_folder_paths (Optional[List[str]], optional): Additional folder paths other than the root where the material's calculations can be found. Defaults to None.
            use_methods_only (bool, optional): When true, some methods can be used without the presence of the DFT calculations directory. This is an internal parameter. Defaults to False.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different atoms. Defaults to None.
            mu_li (float, optional): The chemical potential of Li used to calculate the Li intercalation energies. Defaults to -2.076286119.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present in a material. Defaults to None.
        """
        
        self.material = material
        self.use_methods_only = use_methods_only
        if not use_methods_only:
            if addnl_folder_paths is None:
                os.chdir(material)
            elif addnl_folder_paths is not None:
                addnl_folder_paths = [f"{root}"] + addnl_folder_paths
                addnl_folder_paths = [path+"/" for path in addnl_folder_paths]
                for path in addnl_folder_paths:
                    try:
                        os.chdir(f"{path}"+f"{material}")
                        break
                    except FileNotFoundError:
                        pass
            self.atoms, self.energy = self.get_atoms_and_energy()
            self.structure = AseAtomsAdaptor.get_structure(self.atoms)
            self.formula = str(self.atoms.symbols)
            logging.info(f"Material: {material}, Formula: {self.formula}")
            self.elements = re.findall(r"([A-Z][a-z]?)\d*", self.formula)
            self.metals = [metal for metal in self.elements if metal not in ["O", "S", "C", "N", "Si", "P", "F", "Li"]]
            self.bridging_elements = [element for element in self.elements if element in ["O", "S", "C", "N", "Si", "P", "F"]]
            self.lattice_parameters = list(self.atoms.cell.cellpar()[0:3]/self.atoms.get_volume())
            self.max_void_radius = self.get_max_void_radius(structure=self.structure)
            self.distances = self.get_li_m_b_distances(atoms=self.atoms, custom_cutoffs=custom_cutoffs)
            self.charges = self.get_li_m_b_charges()
            if self.charges==[0,0,0]:
                logging.warning(f"`Bader_calculation` does not exist/is not completed at `{os.getcwd()}`. Taking charges as 0...")
            self.dos_data = self.get_dos_data()
            if self.dos_data==[0]*19:
                logging.warning(f"`Electronic_calculation` does not exist/is not completed at `{os.getcwd()}`. Taking band gap and band centers as 0...")
            self.intercalation_data = self.get_intercalation_data(mu_li=mu_li, custom_n_m = custom_n_m, fhandle=fhandle)
            if self.intercalation_data==[0]*20:
                logging.warning(f"`Intercalation` does not exist/is not completed at `{os.getcwd()}`. Taking Li intercalation features as 0...")
            self.data = [material, self.formula, self.structure] + self.lattice_parameters + [self.max_void_radius] + self.distances + self.charges + self.dos_data + self.intercalation_data
            os.chdir("../")
    
    def get_atoms_and_energy(self, dir_name: str="Energy_calculation") -> Tuple[Atoms, float]:
        """Retrieves the Atoms object and the corresponding energy from the specified directory. Either `OUTCAR` or `vasprun.xml` should be present in the directory.

        Args:
            dir_name (str, optional): Name of the directory. Defaults to "Energy_calculation".

        Returns:
            Tuple[Atoms, float]: ASE Atoms object and the corresponding energy.
        """
        if not os.path.exists(dir_name):
            raise FileNotFoundError(f"{dir_name} does not exist at `{os.getcwd()}`.")
        os.chdir(dir_name)
        try:
            if os.path.exists("OUTCAR"):
                atoms = read("OUTCAR@-1")
            elif os.path.exists("vasprun.xml"):
                atoms = read("vasprun.xml")
            energy = atoms.get_potential_energy()
        except:
            raise IOError(f"Failed to read OUTCAR/vasprun.xml from `{dir_name}` at `{os.getcwd()}`.")
        os.chdir("../")
        return atoms, energy
    
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
    
    def get_max_void_radius(self, structure: Structure) -> float:
        """Provides the maximum void radius in a given structure.

        Args:
            structure (Structure): Pymatgen structure object for which the maximum void radius is to be calculated.

        Returns:
            float: Maximum void radius of the structure.
        """
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
    
    def get_li_m_b_distances(self, atoms: Atoms, custom_cutoffs: Optional[dict]=None) -> List[float]:
        """Calculates the average distances between Li-M, Li-B, and M-B atoms.

        Args:
            atoms (Atoms): ASE Atoms object for which distances are to be calculated.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different elements. Defaults to None.

        Returns:
            List[float]: The average distances [Li-M, Li-B, M-B].
        """
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
        """Computes the average charges on Li, M, and B atoms using Bader analysis from the specified directory. Either `with_charges.traj` or `ACF.dat` should be present in the directory.

        Args:
            dir_name (str, optional): Name of the directory. Defaults to "Bader_calculation".

        Returns:
            List[float]: The average charges of [Li, M, B].
        """
        try:
            os.chdir(dir_name)
            try:
                atoms = read("with_charges.traj")
            except FileNotFoundError:
                try:
                    atoms = get_atoms_with_charges()
                except IOError:
                    os.chdir("../")
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
            os.chdir("../")
            return [li_charge, m_charge, b_charge]
        except FileNotFoundError:
            return [0,0,0]
    
    def get_dos_data(self, dir_name: str="Electronic_calculation", dos: Optional[CompleteDos]=None) -> List[float]:
        """Retrieves the band gap and band centers from the DOS data in the specified directory. Either `DOSCAR` or `vasprun.xml` should be present in the directory.

        Args:
            dir_name (str, optional): Name of the directory. Defaults to "Electronic_calculation".
            dos (Optional[CompleteDos], optional): CompleteDos object from which band gap and band centers are retrieved. This is an internal parameter. Defaults to None.

        Returns:
            List[float]: The band gap and band centers.
        """
        metals = self.metals
        bridging_elements = self.bridging_elements
        use_methods_only = self.use_methods_only
        def get_band_centers(dos: DOS, dos_up: np.ndarray, dos_down: np.ndarray) -> List[float]:
            band_center = dos.get_band_center(dos_up, dos_down)
            val_band_center = dos.get_band_center(dos_up=dos_up, dos_down=dos_down, energy_range=[dos.energies_wrt_fermi[0],0])
            cond_band_center = dos.get_band_center(dos_up=dos_up, dos_down=dos_down, energy_range=[0,dos.energies_wrt_fermi[-1]])
            return [band_center, val_band_center, cond_band_center]
        try:
            if not use_methods_only:
                os.chdir(dir_name)
            if os.path.exists("DOSCAR") and not use_methods_only:
                dos = DOS()
                band_gap = dos.get_band_gap()
                temp_dos_up, temp_dos_down = dos.get_total_dos()
                band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("p")
                p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_orbital_projected_dos("d")
                d_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.atoms if atom.symbol in metals]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                metal_p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"d")
                metal_d_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                indices = [atom.index for atom in self.atoms if atom.symbol in bridging_elements]
                temp_dos_up, temp_dos_down = dos.get_select_atoms_orbital_projected_dos(indices,"p")
                brid_p_band_centers = get_band_centers(dos, temp_dos_up, temp_dos_down)
                os.chdir("../")
                return [band_gap] + band_centers + p_band_centers + d_band_centers + metal_p_band_centers + metal_d_band_centers + brid_p_band_centers
            elif os.path.exists("vasprun.xml") or dos:
                if not use_methods_only:
                    vasprun = Vasprun("vasprun.xml")
                    dos = vasprun.complete_dos
                band_gap = dos.get_gap()
                energies = dos.energies - dos.efermi
                mask_val = energies < 0
                mask_cond = energies >= 0
                try:
                    tdos = dos.densities[Spin.up]+dos.densities[Spin.down]
                except KeyError:
                    tdos = dos.densities[Spin.up]
                band_centers = [np.average(energies, weights=tdos), np.average(energies[mask_val], weights=tdos[mask_val]), np.average(energies[mask_cond], weights=tdos[mask_cond])]
                p_band_centers = [dos.get_band_center(band=OrbitalType.p), dos.get_band_center(band=OrbitalType.p, erange=[min(energies), 0]), dos.get_band_center(band=OrbitalType.p, erange=[0, max(energies)])]
                d_band_centers = [dos.get_band_center(band=OrbitalType.d), dos.get_band_center(band=OrbitalType.d, erange=[min(energies), 0]), dos.get_band_center(band=OrbitalType.d, erange=[0, max(energies)])]
                metal_p_band_centers = [dos.get_band_center(band=OrbitalType.p, elements=metals), dos.get_band_center(band=OrbitalType.p, elements=metals, erange=[min(energies), 0]), dos.get_band_center(band=OrbitalType.p, elements=metals, erange=[0, max(energies)])]
                metal_d_band_centers = [dos.get_band_center(band=OrbitalType.d, elements=metals), dos.get_band_center(band=OrbitalType.d, elements=metals, erange=[min(energies), 0]), dos.get_band_center(band=OrbitalType.d, elements=metals, erange=[0, max(energies)])]
                brid_p_band_centers = [dos.get_band_center(band=OrbitalType.p, elements=bridging_elements), dos.get_band_center(band=OrbitalType.p, elements=bridging_elements, erange=[min(energies), 0]), dos.get_band_center(band=OrbitalType.p, elements=bridging_elements, erange=[0, max(energies)])]
                if not self.use_methods_only:
                    os.chdir("../")
                return [band_gap] + band_centers + p_band_centers + d_band_centers + metal_p_band_centers + metal_d_band_centers + brid_p_band_centers
            else:
                if not use_methods_only:
                    os.chdir("../")
                return [0]*19
        except FileNotFoundError:
            return [0]*19
    
    def get_intercalation_data(self, mu_li: float, custom_n_m: Optional[dict]=None, fhandle: Optional[IO]=None) -> List[float]:
        """Calculates the Li intercalation related properties for the material.

        Args:
            mu_li (float): The chemical potential of Li used to calculate the Li intercalation energies.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present in a material. Defaults to None.
            fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.

        Returns:
            List[float]: Li intercalation features.
        """
        atoms = self.atoms
        energy = self.energy
        volume = atoms.get_volume()
        if custom_n_m is None:
            custom_n_m = {"Li7NbS2": 8}
        try:
            n_m = custom_n_m[self.material]
        except (KeyError, TypeError):
            n_m = sum(1 for atom in atoms if atom.symbol in self.metals)
        if fhandle:
            fhandle.write(f"Material: {self.material}, Formula: {self.formula}:\n")
            str_format = "{:^15} {:^28} {:^15} {:^15} {:^15} {:^23} {:^23} {:^23}\n"
            fhandle.write(str_format.format("Site", "Li Intercalation Energy (eV)", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", "Charge on Li", "Charge on M", "Charge on B"))
        data = {}
        try:
            nlifolders = [entry.name for entry in os.scandir("Intercalation") if entry.is_dir()]
            os.chdir("Intercalation")
            for nlifolder in nlifolders:
                match = re.match(r"(\d+)_Li", nlifolder)
                if match:
                    n_li = int(match.group(1))
                else:
                    n_li = 0
                if not (0.20<=n_li/n_m<=0.30 or 0.45<=n_li/n_m<=0.55):
                    continue
                fhandle.write(f"\tNumber of Li: {n_li}\n") if fhandle else None
                os.chdir(nlifolder)
                oswalk = [i for i in os.walk(".")]
                sites = sorted(oswalk[0][1])
                li_energies, volume_changes, li_m_distances, li_b_distances, m_b_distances, li_charges, m_charges, b_charges, b_val_band_centers, b_cond_band_centers = [], [], [], [], [], [], [], [], [], []
                for site in sites:
                    os.chdir(f"{site}")
                    try:
                        atoms_with_li, energy_with_li = self.get_atoms_and_energy("geo_opt")
                    except (FileNotFoundError, OSError):
                        os.chdir("../")
                        continue
                    volume_with_li = atoms_with_li.get_volume()
                    li_energy = round((energy_with_li-energy-n_li*mu_li)/(n_li),3)
                    volume_change = (volume_with_li - volume)/volume
                    li_m_b_distances = self.get_li_m_b_distances(atoms_with_li)
                    li_m_b_charges = self.get_li_m_b_charges("bader")
                    b_val_cond_band_centers = self.get_dos_data(dir_name="dos")[17:19]
                    fhandle.write(str_format.format(site, li_energy, li_m_b_distances[0], li_m_b_distances[1], li_m_b_distances[2], li_m_b_charges[0], li_m_b_charges[1], li_m_b_charges[2])) if fhandle else None
                    lists = [li_energies, volume_changes, li_m_distances, li_b_distances, m_b_distances, li_charges, m_charges, b_charges, b_val_band_centers, b_cond_band_centers]
                    values = [li_energy, volume_change]+li_m_b_distances+li_m_b_charges+b_val_cond_band_centers
                    for lst, val in zip(lists, values):
                        lst.append(val)
                    os.chdir("../")
                mlei = li_energies.index(min(li_energies))  # mlei: minimum Li energy index
                if (li_charges[mlei]==0 and m_charges[mlei]==0 and b_charges[mlei]==0):
                    logging.warning(f"bader does not exist/is not completed at `{os.getcwd()}/{sites[mlei]}`. Taking charges as 0...")
                if (b_val_band_centers[mlei]==0 and b_cond_band_centers[mlei]==0):
                    logging.warning(f"dos does not exist/is not completed at `{os.getcwd()}/{sites[mlei]}`. Taking band centers as 0...")
                data[round(n_li/n_m,2)] = [li_energies[mlei], volume_changes[mlei], li_m_distances[mlei], li_b_distances[mlei], m_b_distances[mlei], li_charges[mlei], m_charges[mlei], b_charges[mlei], b_val_band_centers[mlei], b_cond_band_centers[mlei]]
                if 0.2<=n_li/n_m<=0.3:
                    data[0.25] = data[round(n_li/n_m,2)]
                elif 0.45<=n_li/n_m<=0.55:
                    data[0.5] = data[round(n_li/n_m,2)]
                os.chdir("../")
            os.chdir("../")
            fhandle.write("\n") if fhandle else None
            try:
                data[0.25]
            except KeyError:
                logging.warning(f"Intercalation data does not exist for 0.25 Li/M at `{os.getcwd()}`. Taking values as 0...")
                data[0.25] = [0]*10
            try:
                data[0.5]
            except KeyError:
                logging.warning(f"Intercalation data does not exist for 0.5 Li/M at `{os.getcwd()}`. Taking values as 0...")
                data[0.5] = [0]*10
            return data[0.25]+data[0.5]
        except FileNotFoundError:
            return [0]*20

def get_material_features(materials: List[str], tag: Optional[str]=None, fhandle: Optional[IO]=None, addnl_folder_paths: Optional[List[str]]=None, mu_li: float=-2.076286119, custom_cutoffs: Optional[dict]=None) -> DataFrame:
    """Extracts features for a list of materials using the GetFeatures class.

    Args:
        materials (List[str]): List of material names. Each material should have a directory with the same name where all DFT calculations are stored.
        tag (Optional[str], optional): Features are saved in a file named `material_features_{tag}.pkl` if `tag` is provided, otherwise in `material_features.pkl`. Defaults to None.
        fhandle (Optional[IO], optional): File handle to write the Li intercalation data. Defaults to None.
        addnl_folder_paths (Optional[List[str]], optional): Additional folder paths other than the root where the material's calculations can be found. Defaults to None.
        mu_li (float, optional): The chemical potential of Li used to calculate the Li intercalation energies. Defaults to -2.076286119.
        custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different elements. Defaults to None.
        
    Returns:
        DataFrame: Features for all materials.
    """
    df = pd.DataFrame(columns=["material", "formula", "structure", "Lattice Parameter a", "Lattice Parameter b", "Lattice Parameter c", "Maximum Void Radius", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", "Charge on Li", "Charge on M", "Charge on B", "Band Gap", "Band Center", "Valence Band Center", "Conduction Band Center", "p Band Center", "Valence p Band Center", "Conduction p Band Center", "d Band Center", "Valence d Band Center", "Conduction d Band Center", "M p Band Center", "M Valence p Band Center", "M Conduction p Band Center", "M d Band Center", "M Valence d Band Center", "M Conduction d Band Center", "B p Band Center", "B Valence p Band Center", "B Conduction p Band Center", "Li Intercalation Energy @ 0.25 Li/M", "Volume Change @ 0.25 Li/M", "Average Li-M Distance @ 0.25 Li/M", "Average Li-B Distance @ 0.25 Li/M", "Average M-B Distance @ 0.25 Li/M", "Charge on Li @ 0.25 Li/M", "Charge on M @ 0.25 Li/M", "Charge on B @ 0.25 Li/M", "B Valence p Band Center @ 0.25 Li/M", "B Conduction p Band Center @ 0.25 Li/M", "Li Intercalation Energy @ 0.50 Li/M", "Volume Change @ 0.50 Li/M", "Average Li-M Distance @ 0.50 Li/M", "Average Li-B Distance @ 0.50 Li/M", "Average M-B Distance @ 0.50 Li/M", "Charge on Li @ 0.50 Li/M", "Charge on M @ 0.50 Li/M", "Charge on B @ 0.50 Li/M", "B Valence p Band Center @ 0.50 Li/M", "B Conduction p Band Center @ 0.50 Li/M"])
    for material in materials:
        features = GetFeatures(material=material, fhandle=fhandle, addnl_folder_paths=addnl_folder_paths, mu_li=mu_li, custom_cutoffs=custom_cutoffs)
        next_index = len(df)
        df.loc[next_index] = features.data
    os.chdir(root)
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