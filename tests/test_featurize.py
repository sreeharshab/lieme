import os
from pandas import DataFrame
from lieme.featurize import get_material_features

def test_featurize():
    root = os.getcwd()
    df = get_material_features(materials=["MoO2"], addnl_folder_paths=[root+"/tests/test_featurize"])
    assert df is not None and isinstance(df, DataFrame)
    os.chdir(root+f"/tests/test_featurize")
    try:
        df = get_material_features(materials=["LiFePO4"])
    except OSError:
        pass
    assert df is not None and isinstance(df, DataFrame)
    os.chdir(root)
    try:
        df = get_material_features(materials=["Nb2O5-T"], addnl_folder_paths=[root+"/tests/test_featurize"])
    except FileNotFoundError:
        pass
    assert df is not None and isinstance(df, DataFrame)
    os.chdir(root)
    df = get_material_features(materials=["TiO2-A"], addnl_folder_paths=[root+"/tests/test_featurize"], custom_cutoffs={"Mn":2, "Co":2, "Fe":2.5, "Nb":2, "C":1.8, "N":1.8})
    assert df is not None and isinstance(df, DataFrame)
    fhandle = open(root+"/tests/test_featurize/Intercalation_data.txt", "w")
    df = get_material_features(materials=["Li3VO4"], addnl_folder_paths=[root+"/tests/test_featurize"], fhandle=fhandle)
    assert df is not None and isinstance(df, DataFrame)
    df = get_material_features(materials=["LiCoO2"], addnl_folder_paths=[root+"/tests/test_featurize"])
    assert df is not None and isinstance(df, DataFrame)