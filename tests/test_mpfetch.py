from lieme.mpfetch import FetchMaterials
import os
from mp_api.client import MPRester
from pandas import DataFrame

def test_mpfetch():
    api_key = os.getenv("MP_API_KEY")
    root = os.getcwd()
    os.chdir(root+"/tests/test_mpfetch/test_1")
    with MPRester(api_key) as mpr:
        results = mpr.materials.summary.search(
            material_ids=["mp-510536", "mp-1221227", "mp-19254", "mp-2291", "mp-17775", "mp-542830"],
            fields=[
                "material_id", "formula_pretty", "composition", "structure", 
                "band_gap", "dos", "formation_energy_per_atom", "is_stable"
            ],
            num_chunks=None,
            chunk_size=1000
        )
    fetcher = FetchMaterials(api_key=api_key)
    X_test = fetcher.get_material_features(results=results, tag="test")
    assert X_test is not None and isinstance(X_test, DataFrame)
    os.chdir(root+"/tests/test_mpfetch/test_2")
    fetcher = FetchMaterials(api_key=api_key)
    def custom_constraint(dummy_param):
        return True
    X_test = fetcher.get_material_features(tag="test", custom_constraints=[(custom_constraint, {"dummy_param": "dummy"})])
    assert X_test is not None and isinstance(X_test, DataFrame)
    os.chdir(root+"/tests/test_mpfetch/test_3")
    fetcher = FetchMaterials(api_key=api_key)
    try:
        X_test = fetcher.get_material_features(tag="test")
    except FileNotFoundError:
        pass
    os.chdir(root)