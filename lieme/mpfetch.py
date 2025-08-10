from lieme.featurize import GetFeatures
from collections import Counter
from typing import List, Tuple, Callable, Dict, Optional
import pandas as pd
from mp_api.client import MPRester
from mp_api.client.core.client import MPRestError
from emmet.core.summary import SummaryDoc
from pymatgen.core.structure import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.ase import AseAtomsAdaptor
from matminer.featurizers.structure import DensityFeatures, MaximumPackingEfficiency
from matminer.featurizers.composition import ElementProperty

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
    
    def follows_constraints(self, composition: Composition, structure: Structure, standard_constraints: bool=True, custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None) -> bool:
        """Checks whether a material follows the specified constraints.

        Args:
            composition (Composition): Pymatgen composition object of the material.
            structure (Structure): Pymatgen structure object of the material.
            standard_constraints (bool, optional): If True, checks whether the material follows the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, checks whether the material follows the custom constraints. Defaults to None.

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
    
    def query_mp(self, standard_constraints: bool=True, custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None) -> List[SummaryDoc]:
        """Queries the Materials Project database for materials that follow the specified constraints.

        Args:
            standard_constraints (bool, optional): If True, checks whether the material follows the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, checks whether the material follows the custom constraints. Defaults to None.

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
            result for result in results if self.follows_constraints(result.composition, result.structure, standard_constraints, custom_constraints)
        ]
        return filtered_results
    
    def get_M_B(self, result: SummaryDoc) -> Tuple[List[str], List[str]]:
        """Extracts the elements, metals, and bridging elements from a material's SummaryDoc object.

        Args:
            result (SummaryDoc): Materials Project SummaryDoc object containing material information.

        Returns:
            Tuple[List[str], List[str]]: Elements, metals and bridging elements of the material.
        """
        elements = [str(el) for el in result.composition.elements]
        metals = [metal for metal in elements if metal not in ["O", "S", "C", "N", "Si", "P", "F", "Li"]]
        bridging_elements = [element for element in elements if element in ["O", "S", "C", "N", "Si", "P", "F"]]
        return elements, metals, bridging_elements

    def get_material_features(self, results: Optional[List[SummaryDoc]]=None, tag: Optional[str]=None, standard_constraints: bool=True, custom_constraints: Optional[List[Tuple[Callable[..., bool], Dict]]]=None, custom_cutoffs: Optional[dict]=None) -> pd.DataFrame:
        """Extracts features from a list of material's SummaryDoc objects.

        Args:
            results (Optional[List[SummaryDoc]], optional): List of SummaryDoc objects containing material information. Defaults to None.
            tag (Optional[str], optional): Features are saved in a file named `material_features_{tag}.pkl` if `tag` is provided, otherwise in `material_features.pkl`. Defaults to None.
            standard_constraints (bool, optional): If True, checks whether the material follows the standard constraints. Defaults to True.
            custom_constraints (Optional[List[Tuple[Callable[..., bool], Dict]]], optional): If not None, checks whether the material follows the custom constraints. Defaults to None.
            custom_cutoffs (Optional[dict], optional): Custom neighbor list cutoffs for different elements. Defaults to None.

        Returns:
            pd.DataFrame: Features for all materials.
        """
        if results is None:
            results = self.query_mp(standard_constraints, custom_constraints) 
        df = pd.DataFrame(columns=["material", "formula", "composition", "structure", "Lattice Parameter a", "Lattice Parameter b", "Lattice Parameter c", "Maximum Void Radius", "Average Li-M Distance", "Average Li-B Distance", "Average M-B Distance", "Band Gap", "Band Center", "Valence Band Center", "Conduction Band Center", "p Band Center", "Valence p Band Center", "Conduction p Band Center", "d Band Center", "Valence d Band Center", "Conduction d Band Center", "M p Band Center", "M Valence p Band Center", "M Conduction p Band Center", "M d Band Center", "M Valence d Band Center", "M Conduction d Band Center", "B p Band Center", "B Valence p Band Center", "B Conduction p Band Center"])
        for result in results:
            features = GetFeatures(str(result.material_id), use_methods_only=True)
            max_void_radius = features.get_max_void_radius(result.structure)
            atoms = AseAtomsAdaptor.get_atoms(result.structure)
            lattice_parameters = list(atoms.cell.cellpar()[0:3]/atoms.get_volume())
            _, features.metals, features.bridging_elements = self.get_M_B(result)
            distances = features.get_Li_M_B_distances(atoms, custom_cutoffs=custom_cutoffs)
            try:
                dos_data = features.get_dos_data(dos=self.mpr.get_dos_by_material_id(result.material_id))
            except MPRestError:
                dos_data = [0]*19
            data = [result.material_id, result.formula_pretty, result.composition, result.structure] + lattice_parameters + [max_void_radius] + distances + dos_data
            next_index = len(df)
            df.loc[next_index] = data
        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        df = ep_feat.featurize_dataframe(df, col_id="composition") 
        df_feat = DensityFeatures()
        df = df_feat.featurize_dataframe(df, col_id="structure")
        mpe_feat = MaximumPackingEfficiency()
        df = mpe_feat.featurize_dataframe(df, col_id="structure")
        file_name = f"material_features_{tag}.pkl" if tag else "material_features.pkl"
        df.to_pickle(file_name)
        return df