from collections import Counter
from typing import List, Tuple, Callable, Dict, Optional
import pandas as pd
from ase.calculators.calculator import Calculator
from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from emmet.core.summary import SummaryDoc
from pymatgen.core.structure import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty
from lieme.featurize import GetFeatures

class FetchMaterials:
    def __init__(self, api_key: str):
        """Initialize the FetchMaterials class to fetch relevant materials from the Materials Project database.

        Args:
            api_key (str): API key to access the Materials Project database.
        """
        self.api_key = api_key
        self.mpr = MPRester(api_key)
        self.composition_space = None
        self.structure_space = None
        self.df_train = None
   
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
                 ) -> List[SummaryDoc]:
        """Queries the Materials Project database for materials that follow the specified constraints.

        Args:
            standard_constraints (bool, optional): If True, checks whether the material follows 
                the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, 
                checks whether the material follows the custom constraints. Defaults to None.

        Returns:
            List[SummaryDoc]: Materials Project SummaryDoc object for each material that follows the specified constraints.
        """
        results = self.mpr.materials.summary.search(
            theoretical=False,
            fields=[
                "material_id", "formula_pretty", "composition", "structure", 
                "band_gap", "dos", "formation_energy_per_atom"
            ],
            num_chunks=None,
            chunk_size=1000
        )
        try:
            self.df_train = pd.read_pickle("material_features_train.pkl")
        except:
            raise FileNotFoundError(
                "The file material_features_train.pkl does not exist.\n"
                "Please run `get_material_features(tag=\"train\")` from `lieme.featurize` to generate it."
            )
        filtered_results = [
            result for result in results if self.follows_constraints(result.composition, result.structure, 
                                                                     standard_constraints, custom_constraints)
        ]
        return filtered_results

    def get_material_features(self, 
                              results: Optional[List[SummaryDoc]]=None, 
                              tag: Optional[str]=None, 
                              standard_constraints: bool=True, 
                              custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None, 
                              custom_cutoffs: Optional[dict]=None, 
                              calc: Optional[Calculator]=None, 
                              fmax: float=0.05, 
                              li_m_ratios: List[float]=[0.25], 
                              mu_li: float=-2.076286119, 
                              custom_n_m: Optional[dict]=None,
                              sampling_size: int=30,
                              seed: int=10,
                              ) -> pd.DataFrame:
        """Extracts features from a list of material's SummaryDoc objects.

        Args:
            results (Optional[List[SummaryDoc]], optional): List of SummaryDoc objects containing 
                material information. Defaults to None.
            tag (Optional[str], optional): Features are saved in a file named `material_features_{tag}.pkl` 
                if `tag` is provided, otherwise in `material_features.pkl`. Defaults to None.
            standard_constraints (bool, optional): If True, checks whether the material follows the 
                standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, 
                checks whether the material follows the custom constraints. Defaults to None.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different elements. Defaults to None.
            calc (Calculator, optional): ASE Calculator object for intercalation calculations. Defaults to None.
            fmax (float, optional): Maximum force criterion for relaxation. Defaults to 0.05 eV/Å.
            li_m_ratios (List[float], optional): List of Li/M ratios for which Li intercalation features 
                are to be calculated. Defaults to [0.25].
            mu_li (float): The chemical potential of Li used to calculate the Li intercalation energies. 
                Defaults to -2.076286119 eV/atom.
            custom_n_m (Optional[dict], optional): Custom number of metal atoms present in a material. Defaults to None.
            sampling_size (int, optional): Number of random intercalated structures to sample. Defaults to 30.
            seed (int, optional): Random seed for sampling intercalated structures. Defaults to 10.

        Returns:
            pd.DataFrame: Features for all materials.
        """
        if results is None:
            results = self.query_mp(standard_constraints, custom_constraints) 
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
        intercalation_cols = intercalation_cols if calc else []
        df = pd.DataFrame(columns=base_cols+intercalation_cols) 
        for result in results:
            material = str(result.material_id)
            atoms = AseAtomsAdaptor.get_atoms(result.structure)
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
                dos_data = features.get_dos_data(dos=self.mpr.get_dos_by_material_id(result.material_id))
            except MPRestError:
                dos_data = [0]*19
            data = ([result.material_id, result.formula_pretty, result.composition, result.structure] 
                    + lattice_parameters + [max_void_radius] + distances + dos_data)
            if calc:
                intercalation_data = features.get_intercalation_data(
                    li_m_ratios=li_m_ratios,
                    mu_li=mu_li,
                    custom_n_m=custom_n_m,
                    sampling_size=sampling_size,
                    seed=seed
                )
                data = data + intercalation_data
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