from lieme.ml import MaterialsEchemRegressor
import os

def test_ml_train():
    root = os.getcwd()
    os.chdir(root+"/tests/test_ml/test_1")
    regressor = MaterialsEchemRegressor()
    regressor.preprocess_data(tag="train")
    regressor.generate_train_jobs()
    regressor.train(TPOT_generations=1, TPOT_population_size=1)
    regressor.write_train_results_to_db()
    regressor.get_train_score_distribution()
    regressor.get_feature_counts()
    predictions = regressor.test(excluded_features=["feature_5"])
    assert predictions is not None
    os.chdir(root+"/tests/test_ml/test_2")
    regressor = MaterialsEchemRegressor()
    try:
        predictions = regressor.test()
    except FileNotFoundError:
        pass
    os.chdir(root+"/tests/test_ml/test_3")
    regressor = MaterialsEchemRegressor()
    try:
        regressor.preprocess_data(tag="train")
    except FileNotFoundError:
        pass
    try:
        regressor.generate_train_jobs()
    except FileNotFoundError:
        pass
    try:
        regressor.train(TPOT_generations=1, TPOT_population_size=1)
    except FileNotFoundError:
        pass
    os.chdir(root+"/tests/test_ml/test_4")
    regressor = MaterialsEchemRegressor()
    regressor.preprocess_data(tag="train")
    regressor.generate_train_jobs(exclude_jobs=[0,1,2,3])
    regressor.generate_train_jobs(exclude_jobs=[["Band Gap", "Li/M", "feature_4", "feature_3"]])
    try:
        regressor.train(TPOT_generations=1, TPOT_population_size=1)
    except FileNotFoundError:
        pass
    os.chdir(root+"/tests/test_ml/test_5")
    regressor = MaterialsEchemRegressor()
    regressor.preprocess_data(tag="train")
    try:
        regressor.train(TPOT_generations=1, TPOT_population_size=1)
    except FileNotFoundError:
        pass
    os.chdir(root)