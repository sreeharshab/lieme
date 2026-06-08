import warnings
import contextlib
import io
import gc
from tqdm import tqdm
from collections import Counter
from typing import List, Tuple, Callable, Dict, Optional
import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator
from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from jarvis.db.figshare import data
from jarvis.core.atoms import Atoms as JarvisAtoms
from qmpy_rester import QMPYRester
from pymatgen.core.structure import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from lieme.featurize import GetFeatures

class Material:
    def __init__(self, material_id: str, atoms: Atoms=None, structure: Structure=None, **kwargs):
        self.material_id = material_id
        self.atoms = atoms
        self.structure = structure
        if structure is None and atoms is not None:
            self.structure = AseAtomsAdaptor.get_structure(atoms)
        elif atoms is None and structure is not None:
            self.atoms = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise ValueError("Either atoms or structure must be provided.")
        if kwargs.get("composition", None):
            self.composition = kwargs["composition"]
        else:
            self.composition = structure.composition
        if kwargs.get("formula_pretty", None):
            self.formula_pretty = kwargs["formula_pretty"]
        else:
            self.formula_pretty = self.composition.reduced_formula
        self.dos = kwargs.get("dos", None)
        self.band_gap = kwargs.get("band_gap", None)
    
    def __str__(self):
        return (f"Material ID: {self.material_id}\n"
                f"Formula: {self.formula_pretty}\n"
                f"Structure: {self.structure}\n"
                )
    
    def __repr__(self):
        return self.__str__()

class FetchMaterials:
    def __init__(self, api_key: str, jarvis_json: str=None):
        """Initialize the FetchMaterials class to fetch relevant materials from the Materials Project database.

        Args:
            api_key (str): API key to access the Materials Project database.
            jarvis_json (str, optional): Path to the Jarvis JSON file. Defaults to None.
        """
        self.api_key = api_key
        self.jarvis_json = jarvis_json
        self.mpr = MPRester(api_key)
        self.qmpyr = QMPYRester()
        self.composition_space = None
        self.structure_space = None
        try:
            self.df_train = pd.read_pickle("material_features_train.pkl")
        except:
            warnings.warn("The file material_features_train.pkl does not exist.\n"
                          "Please run `get_material_features(tag=\"train\")` from" 
                          "`lieme.featurize` to generate it.\nOr switch standard_constraints "
                          "to False while querying materials.",
                          UserWarning
            )
   
    def get_composition_space(self) -> List[str]:
        """Provides the composition space of the training data.

        Returns:
            List[str]: Composition space as a list of elements.
        """
        if self.composition_space is not None:
            return self.composition_space
        composition_space = Counter()
        for composition in self.df_train["composition"]:
            composition_space.update(composition.elements)
        self.composition_space = sorted(set(str(el) for el in composition_space))
        return self.composition_space

    def get_structure_space(self) -> List[str]:
        """Provides the structure space of the training data.

        Returns:
            List[str]: Structure space as a list of space groups.
        """
        if self.structure_space is not None:
            return self.structure_space
        structure_space = set()
        for structure in self.df_train["structure"]:
            sga = SpacegroupAnalyzer(structure, symprec=1e-1)
            structure_space.add(sga.get_space_group_symbol())
        self.structure_space = sorted(structure_space)
        return self.structure_space
    
    def apply_standard_constraints(self, composition: Composition, structure: Structure) -> bool:
        """Checks whether a material follows the standard constraints.

        Args:
            composition (Composition): Pymatgen composition object of the material.
            structure (Structure): Pymatgen structure object of the material.

        Returns:
            bool: True if the material follows the standard constraints, False otherwise.
        """
        composition_space = self.get_composition_space()
        metals_space = set([metal for metal in composition_space if metal not in ["O", "S", "C", "N", "Si", "P", "F", "Li"]])
        bridging_elements_space = set(["O", "S", "C", "N", "Si", "P", "F"])
        elements = set(str(el) for el in composition.elements)
        metals = elements.intersection(metals_space)
        bridging_elements = elements.intersection(bridging_elements_space)
        if not elements.issubset(composition_space):
            return False
        if len(metals)<1:
            return False
        if len(metals)>3:
            return False
        if not ({"S", "O", "C"} & bridging_elements):
            return False
        if ("C" in bridging_elements and "N" not in bridging_elements):
            return False
        if ("Si" in bridging_elements or "P" in bridging_elements) and "O" not in bridging_elements:
            return False
        if len(bridging_elements)>2 and set(bridging_elements)!={"S", "O", "F"}:
            return False
        structure_space = self.get_structure_space()
        if SpacegroupAnalyzer(structure).get_space_group_symbol() not in structure_space:
            return False
        return True
    
    def follows_constraints(self, 
                            composition: Composition, 
                            structure: Structure, 
                            standard_constraints: bool=True, 
                            custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None
                            ) -> bool:
        """Checks whether a material follows the specified constraints.

        Args:
            composition (Composition): Pymatgen composition object of the material.
            structure (Structure): Pymatgen structure object of the material.
            standard_constraints (bool, optional): If True, checks whether the material follows 
                the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, 
                checks whether the material follows the custom constraints. Defaults to None.

        Returns:
            bool: True if the material follows the specified constraints, False otherwise.
        """
        if standard_constraints:
            check = self.apply_standard_constraints(composition, structure)
        if custom_constraints:
            for constraint, kwargs in custom_constraints:
                if not constraint(**kwargs):
                    return False
        return check if standard_constraints else True
    
    def query_mp(self, 
                 standard_constraints: bool=True, 
                 custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None
                 ) -> List[Material]:
        """Queries the Materials Project database for materials that follow the specified constraints.

        Args:
            standard_constraints (bool, optional): If True, checks whether the material follows 
                the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, 
                checks whether the material follows the custom constraints. Defaults to None.

        Returns:
            List[Material]: Material object for each material that follows the specified constraints.
        """
        docs = self.mpr.materials.summary.search(
            theoretical=False,
            fields=[
                "material_id", "formula_pretty", "composition", "structure", 
                "band_gap", "dos", "formation_energy_per_atom"
            ],
            num_chunks=None,
            chunk_size=1000
        )
        filtered_results = []
        for i, doc in enumerate(tqdm(docs, desc="Filtering MP materials")):
            if self.follows_constraints(doc.composition, doc.structure, 
                                        standard_constraints, custom_constraints):
                try:
                    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
                        dos = self.mpr.get_dos_by_material_id(doc.material_id)
                except MPRestError:
                    dos = [np.nan]*19
                result = Material(
                    material_id=doc.material_id,
                    structure=doc.structure,
                    composition=doc.composition,
                    formula_pretty=doc.formula_pretty,
                    dos = dos,
                    band_gap=doc.band_gap
                )
                filtered_results.append(result)
            if i%1000==0:
                gc.collect()
        return filtered_results
    
    def query_jarvis(self,
                      standard_constraints: bool=True,
                      custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None
                      ) -> List[Material]:
        """Queries the Jarvis database for materials that follow the specified constraints.
        Args:
            standard_constraints (bool, optional): If True, checks whether the material follows
                the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None,
                checks whether the material follows the custom constraints. Defaults to None.
        
        Returns:
            List[Material]: Material object for each material that follows the specified constraints
        """
        if self.jarvis_json:
            dft_3d = data("dft_3d", store_dir=self.jarvis_json)
        elif not self.jarvis_json:
            try:
                dft_3d = data("dft_3d")
            except Exception as e:
                warnings.warn(f"Could not load JARVIS data.\n{e}", UserWarning)
                return []
        dft_3d = {entry["jid"]: entry for entry in dft_3d}
        filtered_results = []
        for jid, entry in tqdm(dft_3d.items(), desc="Filtering Jarvis materials"):
            structure = JarvisAtoms.from_dict(entry["atoms"]).pymatgen_converter()
            result = Material(
                material_id=jid,
                structure=structure,
                band_gap=entry.get("optb88vdw_bandgap", None)
            )
            if self.follows_constraints(result.composition, result.structure,
                                        standard_constraints, custom_constraints):
                filtered_results.append(result)
        return filtered_results
    
    def query_oqmd(self,
                   standard_constraints: bool=True,
                   custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None
                   ) -> List[Material]:
        entries = []
        chunk_size = 500
        offset = 0
        pbar = tqdm(desc="Fetching OQMD materials")
        while True:
            kwargs = {
                "stability": "<0.1",
                "limit": chunk_size,
                "offset": offset
            }
            phases = self.qmpyr.get_oqmd_phases(verbose=False, **kwargs)
            if not phases:
                break
            entries.extend(phases["data"])
            pbar.update(1)
            offset += chunk_size
        pbar.close()
        filtered_results = []
        for entry in tqdm(entries, desc="Filtering OQMD materials"):
            lattice = entry["unit_cell"]
            sites = entry["sites"]
            species = []
            coords = []
            for site in sites:
                parts = site.split(" @ ")
                species.append(parts[0])
                coords.append([float(x) for x in parts[1].split()])
            structure = Structure(lattice, species, coords)
            result = Material(
                material_id=entry["entry_id"],
                structure=structure,
                band_gap=entry.get("band_gap", None)
            )
            if self.follows_constraints(result.composition, result.structure,
                                        standard_constraints, custom_constraints):
                filtered_results.append(result)
        return filtered_results

    def get_material_features(self, 
                              results: Optional[List[Material]]=None, 
                              standard_constraints: bool=True, 
                              custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None, 
                              custom_cutoffs: Optional[dict]=None,
                              energy_range: Optional[List[float]]=None, 
                              plot_dos: bool=False,
                              calc: Optional[Calculator]=None, 
                              fmax: float=0.05, 
                              li_m_ratios: List[float]=[0.25], 
                              li_m_ratio_tol: float=0.1,
                              mu_li: float=-2.076286119, 
                              custom_n_m: Optional[dict]=None,
                              sampling_size: int=30,
                              seed: int=10,
                              li_atom_cutoff: float=1.7,
                              li_li_cutoff: float=1.0,
                              temperatures: List[float] = [1000, 1500],
                              msd_col_idx: int=7,
                              dt: float=0.2,
                              com: bool=False,
                              interpolate_arrhenius: bool=False,
                              interpolation_temperatures: List[float]=None,
                              plot_diffusion: bool=False,
                              addnl_anions: Optional[dict]=None,
                              mp: bool=True,
                              xc: str="GGA_GGA+U",
                              tag: Optional[str]=None, 
                              ) -> pd.DataFrame:
        """Extracts features from a list of material's SummaryDoc objects.

        Args:
            results (Optional[List[Material]], optional): List of Material objects containing 
                material information. Defaults to None.
            tag (Optional[str], optional): Features are saved in a file named `material_features_{tag}.pkl` 
                if `tag` is provided, otherwise in `material_features.pkl`. Defaults to None.
            standard_constraints (bool, optional): If True, checks whether the material follows the 
                standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, 
                checks whether the material follows the custom constraints. Defaults to None.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different elements. 
                Defaults to None.
            energy_range (Optional[List[float]], optional): Custom energy range wrt fermi level to calculate 
                band centers. Defaults to None.
            calc (Calculator, optional): ASE Calculator object for intercalation calculations. Defaults to None.
            fmax (float, optional): Maximum force criterion for relaxation. Defaults to 0.05 eV/Å.
            li_m_ratios (List[float], optional): List of Li/M ratios for which Li intercalation features 
                are to be calculated. Defaults to [0.25].
            li_m_ratio_tol (float, optional): Tolerance for Li/M ratio matching when reading from existing
                intercalation directories. Defaults to 0.1.
            mu_li (float): The chemical potential of Li used to calculate the Li intercalation energies. 
                Defaults to -2.076286119 eV/atom.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present in a material. Defaults to None.
            sampling_size (int, optional): Number of random intercalated structures to sample. Defaults to 30.
            seed (int, optional): Random seed for sampling intercalated structures. Defaults to 10.
            li_atom_cutoff (float, optional): Li-atom cutoff distance in intercalated structures. Defaults to 1.7 Å.
            li_li_cutoff (float, optional): Li-Li cutoff distance in intercalated structures. Defaults to 1.0 Å.
            addnl_anions (Optional[dict], optional): Additional anions other than the default ones to be 
                considered during decomposition. Defaults to None.
            mp (bool, optional): Whether to obtain the energy of the material itself from Materials Project. 
                If False, the energy of the material is obtained using GetFeatures.energy. Defaults to True.
            xc (str, optional): Exchange-correlation functional used to calculate the energy of the material 
                in Materials Project. Defaults to "GGA_GGA+U".

        Returns:
            pd.DataFrame: Features for all materials.
        """
        if results is None:
            results_mp = self.query_mp(standard_constraints, custom_constraints)
            try:
                results_jarvis = self.query_jarvis(standard_constraints, custom_constraints)
            except:
                results_jarvis = []
            try:
                # results_oqmd = self.query_oqmd(standard_constraints, custom_constraints)
                results_oqmd = []
            except:
                results_oqmd = []
            results = results_mp + results_jarvis + results_oqmd
        base_cols = [
            "material", "formula", "composition", "structure",
            "Lattice Parameter a", "Lattice Parameter b", "Lattice Parameter c",
            "Maximum Void Radius", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance",
            "Band Gap", "Band Center", "Valence Band Center", "Conduction Band Center",
            "p Band Center", "Valence p Band Center", "Conduction p Band Center",
            "d Band Center", "Valence d Band Center", "Conduction d Band Center",
            "M p Band Center", "M Valence p Band Center", "M Conduction p Band Center",
            "M d Band Center", "M Valence d Band Center", "M Conduction d Band Center",
            "B p Band Center", "B Valence p Band Center", "B Conduction p Band Center"
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
        diffusion_cols = []
        for ratio in li_m_ratios:
            for temperature in temperatures:
                diffusion_cols.append(f"Li Diffusion Coefficient @ {ratio:.2f} Li/M and {temperature} K")
        stability_cols = ["Intercalation Stability"]
        df = pd.DataFrame(columns=base_cols+intercalation_cols+diffusion_cols+stability_cols) 
        for result in results:
            material = str(result.material_id)
            atoms = result.atoms
            features = GetFeatures(material=material, 
                                   atoms=atoms, 
                                   calc=calc, 
                                   fmax=fmax,
                                   )
            relaxed_atoms = features.relaxed_atoms
            max_void_radius = features.get_max_void_radius()
            lattice_parameters = list(relaxed_atoms.cell.cellpar()[0:3]/relaxed_atoms.get_volume())
            distances = features.get_li_m_b_distances(custom_cutoffs=custom_cutoffs)
            try:
                dos_data = features.get_dos_data(dos=result.dos, energy_range=energy_range, plot=plot_dos)
            except MPRestError:
                dos_data = [np.nan]*19
            data = ([result.material_id, result.formula_pretty, result.composition, result.structure] 
                    + lattice_parameters + [max_void_radius] + distances + [result.band_gap] + dos_data[1:])
            intercalation_data = features.get_intercalation_data(
                li_m_ratios=li_m_ratios,
                li_m_ratio_tol=li_m_ratio_tol,
                mu_li=mu_li,
                custom_n_m=custom_n_m,
                sampling_size=sampling_size,
                seed=seed,
                li_atom_cutoff=li_atom_cutoff,
                li_li_cutoff=li_li_cutoff
            )
            data = data + intercalation_data
            diffusion_data = []
            diff_coeffs = features.get_diffusion_data(
                li_m_ratios=li_m_ratios,
                temperatures=temperatures,
                msd_col_idx=msd_col_idx,
                dt=dt,
                com=com,
                interpolate_arrhenius=interpolate_arrhenius,
                interpolation_temperatures=interpolation_temperatures,
                plot=plot_diffusion
            )
            for ratio in li_m_ratios:
                for temperature in temperatures:
                    diffusion_data.append(diff_coeffs.get((ratio, temperature), np.nan))
            data = data + diffusion_data
            if material.startswith("mp-"):
                decomposition_data = features.get_intercalation_stability(api_key=self.api_key, 
                                                                            addnl_anions=addnl_anions,
                                                                            mp=mp, 
                                                                            xc=xc)
                data = data + [decomposition_data]
            else:
                data = data + [np.nan]
            next_index = len(df)
            df.loc[next_index] = data
            features.return_to_root()
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        df = ep_feat.featurize_dataframe(df, col_id="composition") 
        df_feat = DensityFeatures()
        df = df_feat.featurize_dataframe(df, col_id="structure")
        mpe_feat = MaximumPackingEfficiency()
        df = mpe_feat.featurize_dataframe(df, col_id="structure")
        file_name = f"material_features_{tag}.pkl" if tag else "material_features.pkl"
        df.to_pickle(file_name)
        return df