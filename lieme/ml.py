import os
import glob
import pickle, json
import sqlite3
from typing import Tuple, List, Optional
import logging
from itertools import combinations
from packaging import version
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn.metrics import get_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tpot
from collections import Counter
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MaterialsEchemRegressor:
    def __init__(self, 
                 jobs_path: str="jobs.pkl",
                 results_dir: str="results",
                 db_path: str="results.db"
                 ):
        """Initializes the MaterialsEchemRegressor to regress the input features to electrochemical performance output.

        Args:
            jobs_path (str, optional): Path to the `jobs.pkl` file which contains the tuples of feature subsets corresponding to the training jobs. Defaults to "jobs.pkl".
            results_dir (str, optional): Directory path to store the training results. Defaults to "results".
            db_path (str, optional): SQLite database path to store the training results (models and metadata). Defaults to "results.db".
        """
        self.jobs_path = jobs_path
        self.results_dir = results_dir
        self.db_path = db_path
        self.models = None
        self.metadata = None
        self.train_results_processed = False

    def preprocess_data(self, 
                        tag: Optional[str]=None, 
                        X: Optional[DataFrame]=None, 
                        Y: Optional[DataFrame]=None, 
                        excluded_materials: List[str]=["MoS2","VO2-M","VO2-R"], 
                        max_correlation_allowed: float=0.8, 
                        PCA_n_components: int=2, 
                        PCA_n_features: int=33
                        ) -> Tuple[DataFrame, DataFrame]:
        """Preprocesses the material features and electrochemical performance data.

        Args:
            tag (Optional[str], optional): Features are preprocessed from a file with the name `material_features_{tag}.pkl` if `tag` is provided, otherwise from `material_features.pkl`. Defaults to None.
            X (Optional[DataFrame], optional): If `X` is None, features are preprocessed from `material_features.pkl`. Defaults to None.
            Y (Optional[DataFrame], optional): If `Y` is None, electrochemical performance is preprocessed from `exp_data.xlsx`. Defaults to None.
            excluded_materials (List[str], optional): Materials present in `material_features.pkl` to be excluded in preprocessing. Defaults to ["MoS2","VO2-M","VO2-R"].
            max_correlation_allowed (float, optional): Correlation cutoff to omit highly correlated features. Defaults to 0.8.
            PCA_n_components (int, optional): Number of components in the PCA analysis. Defaults to 2.
            PCA_n_features (int, optional): Number of top features to be considered according to PCA. Defaults to 33.

        Returns:
            Tuple[DataFrame, DataFrame]: Processed `X` and `Y` which can be used to train the models.
        """
        if X is None:
            file_name = f"material_features_{tag}.pkl" if tag else "material_features.pkl"
            try:
                X = pd.read_pickle(file_name)
            except:
                raise FileNotFoundError(
                f"`X` is not provided and `{file_name}` does not exist.\n"
                f"Please run `get_material_features(tag={tag})` from `lieme.featurize` or `lieme.mpfetch` to generate the file or provide `X` explicitly."
            )
        materials = X["material"].tolist()
        X = X[[material not in excluded_materials for material in materials]]
        X = X.reset_index(drop=True)
        X.loc[X["material"]=="Nb2O5-T", "Band Gap"] = 1.925
        X.loc[X["material"]=="Nb2O5-TT", "Band Gap"] = 1.773
        X = X.drop(columns=["material", "formula", "structure", "composition"])
        X = X.drop([col for col in X.columns if "0.5" in col], axis=1)
        X = X.loc[:, (X!=0).any(axis=0)]
        if len(X)>10:
            X = X.loc[:, X.nunique() > 10]

        corr = X.corr(method="pearson").fillna(0)
        corr = corr.stack().reset_index()
        corr.columns = ["Feature1", "Feature2", "Correlation"]
        corr = corr[corr["Feature1"] < corr["Feature2"]]
        high_corr_pairs = corr[abs(corr["Correlation"]) > max_correlation_allowed]
        high_corr_pairs = high_corr_pairs.sort_values(by="Correlation", ascending=False)
        high_corr_pairs = high_corr_pairs.reset_index(drop=True)
        feature2 = high_corr_pairs["Feature2"].values

        X = X.drop(columns=np.unique(feature2))
        X_arr = X.values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)

        pca = PCA(n_components=PCA_n_components)
        pca.fit(X_scaled)
        loadings = np.abs(pca.components_.T)
        explained_variance_ratio = pca.explained_variance_ratio_
        feature_importance = np.dot(loadings, explained_variance_ratio)
        feature_ranking_indices = np.argsort(-feature_importance)
        selected_features_PCA = X.columns[feature_ranking_indices].values
        X = X[selected_features_PCA[0:PCA_n_features]]
        file_name = f"X_{tag}.pkl" if tag else "X.pkl"
        X.to_pickle(file_name)

        if Y is None and os.path.exists("exp_data.xlsx"):
            Y = pd.read_excel("exp_data.xlsx", header=0)
        elif Y is None and not os.path.exists("exp_data.xlsx"):
            logging.info("`Y` is not provided and `exp_data.xlsx` does not exist. Returning `X` only...")
            return X, None
        Y = Y[["System", "Rev. Cap at 0.1C, mAh/g", "Rev. Cap at 5C, mAh/g"]]
        Y = Y.set_index("System")
        Y = Y.loc[materials, ["Rev. Cap at 0.1C, mAh/g", "Rev. Cap at 5C, mAh/g"]]
        Y = Y[[material not in excluded_materials for material in materials]]
        Y["capacity_ratio"] = (Y["Rev. Cap at 0.1C, mAh/g"].values - Y["Rev. Cap at 5C, mAh/g"].values)/Y["Rev. Cap at 0.1C, mAh/g"].values
        Y = Y["capacity_ratio"]
        file_name = f"Y_{tag}.pkl" if tag else "Y.pkl"
        Y.to_pickle(file_name)

        return X, Y
    
    def generate_train_jobs(self, X_train: Optional[DataFrame]=None, n_features: int=4, exclude_jobs: Optional[List]=None):
        """Generates the training jobs by creating combinations of features to be used for training the models.

        Args:
            X_train (Optional[DataFrame], optional): Processed `X`. Defaults to None.
            n_features (int, optional): Size of the feature subset. Defaults to 4. 
            exclude_jobs (Optional[List], optional): The tuple or index of feature subsets to be excluded. For example, [("Band Gap", "Volume"), ("Volume", "Band Center")] or [4,10]. Defaults to None.
        """
        if X_train is None:
            try:
                X_train = pd.read_pickle("X_train.pkl")
            except:
                raise FileNotFoundError(
                f"`X_train` is not provided and `X_train.pkl` does not exist.\n"
                "Please run `preprocess_data(tag=\"train\")` or provide `X_train` explicitly."
            )
        column_combinations = list(combinations(X_train.columns,n_features))
        if isinstance(exclude_jobs, list) and (all(isinstance(c, tuple) for c in exclude_jobs) or all(isinstance(c, list) for c in exclude_jobs)):
            exclude_jobs = set(frozenset(c) for c in exclude_jobs)
            column_combinations = [c for c in column_combinations if frozenset(c) not in exclude_jobs]
        if isinstance(exclude_jobs, list) and all(isinstance(c, int) for c in exclude_jobs):
            column_combinations = [c for i, c in enumerate(column_combinations) if i not in exclude_jobs]
        with open(self.jobs_path, "wb") as f:
            pickle.dump(column_combinations, f)
    
    def write_train_results(self, job_id: int, model: Pipeline, features: List[str], score: float):
        """Writes the training results to the results directory in a pickle file.
        
        Args:
            job_id (int): ID of the job to be written.
            model (Pipeline): Trained model to be written.
            features (List[str]): Features used to train the model.
            score (float): Cross-validation score of the model.
        """
        results_dir = self.results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        results = {
            "job_id": job_id,
            "model": model,
            "features": features,
            "cv_score": score
        }
        results_path = os.path.join(results_dir, f"results_{job_id}.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
    
    def write_train_results_to_db(self, job_ids: Optional[List[int]]=None, cv_score_cutoff: float=0.5, batch_size: int=100):
        """Writes the training results from the results directory into the SQLite database.

        Args:
            job_ids (Optional[List[int]], optional): List of job IDs to be written. If None, all results files in the `results` directory are written. Defaults to None.
            cv_score_cutoff (float, optional): CV score cutoff to determine top models which are to be written. Note: Metadata will be written for all models. Defaults to 0.5.
            batch_size (int, optional): Number of results to be written in a single batch. Defaults to 100.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=600)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS models (
                job_id INTEGER PRIMARY KEY,
                model BLOB,
                features TEXT,
                cv_score REAL
            )""")
            results_paths = glob.glob(f"{self.results_dir}/results_*.pkl")
            if job_ids:
                results_paths = [path for path in results_paths if int(path.split("_")[-1].split(".")[0]) in job_ids]
            batch_data = []
            for path in results_paths:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if data["cv_score"]>cv_score_cutoff:
                    batch_data.append((
                        data["job_id"],
                        pickle.dumps(data["model"]),
                        json.dumps(data["features"]),
                        data["cv_score"]
                    ))
                else:
                    batch_data.append((
                        data["job_id"],
                        pickle.dumps(None),
                        json.dumps(data["features"]),
                        data["cv_score"]
                    ))
                if len(batch_data)>=batch_size:
                    cursor.executemany(
                        "REPLACE INTO models (job_id, model, features, cv_score) VALUES (?, ?, ?, ?)",
                        batch_data
                    )
                    conn.commit()
                    batch_data = []     
            if batch_data:
                cursor.executemany(
                    "REPLACE INTO models (job_id, model, features, cv_score) VALUES (?, ?, ?, ?)",
                    batch_data
                )
                conn.commit()
        except Exception as e:
            logging.error(f"Cannot write results into database due to the following error.\n{e}")
        finally:
            if conn:
                conn.close()
    
    def train(
            self, 
            X_train: Optional[DataFrame]=None, 
            Y_train: Optional[DataFrame]=None, 
            job_id: int=0,
            TPOT_generations: int=50,
            TPOT_population_size: int=50,
            ) -> Tuple[Pipeline, float]:
        """Trains a model using the specified job_id in reference to the jobs in `jobs.pkl`. Training is done using TPOT.

        Args:
            X_train (Optional[DataFrame], optional): Input material features. If None, `X_train.pkl` should be present in the working directory. Defaults to None.
            Y_train (Optional[DataFrame], optional): Output electrochemical performance. If None, `Y_train.pkl` should be present in the working directory. Defaults to None.
            job_id (int, optional): ID of the job to be trained. Defaults to 0.
            TPOT_generations (int, optional): Number of TPOT generations. Refer to documentation of TPOT for more details. Defaults to 50.
            TPOT_population_size (int, optional): Population size for TPOT. Refer to documentation of TPOT for more details. Defaults to 50.

        Returns:
            Tuple[Pipeline, float]: The model and its cross-validation score.
        """
        if X_train is None:
            try:
                X_train = pd.read_pickle("X_train.pkl")
            except:
                raise FileNotFoundError(
                f"`X_train` is not provided and `X_train.pkl` does not exist.\n"
                "Please run `preprocess_data(tag=\"train\")` or provide `X_train` explicitly."
            )
        if Y_train is None:
            try:
                Y_train = pd.read_pickle("Y_train.pkl")
            except:
                raise FileNotFoundError(
                f"`Y_train` is not provided and `Y_train.pkl` does not exist.\n"
                "Please run `preprocess_data(tag=\"train\")` or provide `Y_train` explicitly."
            )
        try:
            with open(self.jobs_path, "rb") as f:
                all_combinations = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"`jobs.pkl` not found. Please run `generate_train_jobs()` to generate the jobs."
            )
        features = all_combinations[job_id]
        X_train_subset = X_train[list(features)]
        if version.parse(tpot.__version__) <= version.parse("0.12.2"):
            est = tpot.TPOTRegressor(generations=TPOT_generations, population_size=TPOT_population_size, verbosity=2, random_state=42, scoring="r2")
        else:
            est = tpot.TPOTEstimator(search_space="linear", scorers=["r2"], scorers_weights=[1], classification=False, generations=TPOT_generations, population_size=TPOT_population_size, verbose=2, random_state=42, max_time_mins=None)
        try:
            logging.info(f"Training job_id {job_id}...")
            est.fit(X_train_subset.values, Y_train.values)
            model = est.fitted_pipeline_
            if version.parse(tpot.__version__) <= version.parse("0.12.2"):
                score = est._optimized_pipeline_score
            else:
                scorer = get_scorer('r2')
                score = scorer(est, X_train_subset.values, Y_train.values)
        except Exception as e:
            logging.error(f"Cannot train model for job_id {job_id} due to the following error.\n{e}")
            return
        self.write_train_results(job_id, model, features, score)
        return model, score
    
    def process_train_results(self):
        """Processes the training results from the SQLite database and stores them in `self.metadata` and `self.models`.
        """
        if self.train_results_processed:
            return
        conn = sqlite3.connect(self.db_path, timeout=30)
        cursor = conn.cursor()
        cursor.execute("SELECT job_id, model, features, cv_score FROM models")
        rows = cursor.fetchall()
        conn.close()
        job_ids = []
        features = []
        cv_scores = []
        models = {}
        for row in rows:
            job_id, model_blob, features_json, cv_score = row
            job_ids.append(job_id)
            features.append(json.loads(features_json))
            cv_scores.append(cv_score)
            models[job_id] = pickle.loads(model_blob)
        self.metadata = pd.DataFrame({
            "job_id": job_ids,
            "features": features,
            "cv_score": cv_scores
        })
        self.models = models
        self.train_results_processed = True
    
    def get_train_score_distribution(self, 
                                    bins: List[float]=[-np.inf, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1], 
                                    labels: List[str]=["≤0", "0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", ">0.5"]
                                    ) -> plt.Figure:
        """Generates a histogram showing the distribution of train cross-validation scores across models.

        Args:
            bins (List[float], optional): Bins for the histogram. Defaults to [-np.inf, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 1].
            labels (List[str], optional): Labels for the histogram bins. Defaults to ["≤0", "0–0.1", "0.1–0.2", "0.2–0.3", "0.3–0.4", "0.4–0.5", ">0.5"].

        Returns:
            plt.Figure: Matplotlib figure object showing the histogram of train cross-validation scores.
        """
        self.process_train_results()
        metadata = self.metadata
        metadata["score_bin"] = pd.cut(metadata["cv_score"], bins=bins, labels=labels, right=True)
        bin_counts = metadata["score_bin"].value_counts().sort_index()
        fig = plt.figure(dpi = 200, figsize=(6,6))
        ax = fig.gca()
        ax.bar(bin_counts.index, bin_counts.values, color="darkorange")
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(bottom = True, top = True, left = True, right = True)
        ax.tick_params(axis = "x", direction = "in")
        ax.tick_params(axis = "y", direction = "in")
        plt.xlabel("Train CV score range")
        plt.ylabel("Distribution of Train CV score across models")
        plt.savefig(f"train_score_distribution.png", bbox_inches='tight')
        return fig
    
    def get_feature_counts(self, cv_score_cutoff: float=0.5) -> plt.Figure:
        """Generates a bar plot showing the frequency of features in the top models based on the cross-validation score.

        Args:
            cv_score_cutoff (float, optional): CV score cutoff to determine top models. Defaults to 0.5.

        Returns:
            plt.Figure: Matplotlib figure object showing the frequency of features in the top models.
        """
        self.process_train_results()
        metadata = self.metadata
        top_models_info = metadata[metadata["cv_score"] > cv_score_cutoff]
        features = top_models_info["features"].tolist()
        features = [list(i) for i in features]
        features = sum(features, [])
        feature_counts = Counter(features)
        compositional_features = [feature for feature in features if "MagpieData" in feature]
        electronic_features = [feature for feature in features if ("Charge" in feature or "Band Center" in feature or "Band Gap" in feature) and ("Li/M" not in feature)]
        intercalation_features = [feature for feature in features if "Li/M" in feature]
        structural_features = [feature for feature in features if feature not in (compositional_features + electronic_features + intercalation_features)]
        grouped_features = {
            "compositional": [],
            "structural": [],
            "electronic": [],
            "intercalation": []
        }
        for feature, count in feature_counts.items():
            if feature in compositional_features:
                grouped_features["compositional"].append((feature, count))
            elif feature in structural_features:
                grouped_features["structural"].append((feature, count))
            elif feature in electronic_features:
                grouped_features["electronic"].append((feature, count))
            elif feature in intercalation_features:
                grouped_features["intercalation"].append((feature, count))
        for key in grouped_features:
            grouped_features[key] = sorted(grouped_features[key], key=lambda x: x[1])
        ordered_features = (
            grouped_features["compositional"] +
            grouped_features["structural"] +
            grouped_features["electronic"] +
            grouped_features["intercalation"]
        )
        a = [f[0] for f in ordered_features]
        a_modified = [f[0].replace("MagpieData ", "") for f in ordered_features]
        b = [f[1] for f in ordered_features]
        c = list(range(len(a)))
        feature_color_map = {
            "compositional": "tomato",
            "structural": "skyblue",
            "electronic": "yellowgreen",
            "intercalation": "darkorchid"
        }
        colors = []
        for feature in a:
            if feature in compositional_features:
                colors.append(feature_color_map["compositional"])
            elif feature in structural_features:
                colors.append(feature_color_map["structural"])
            elif feature in electronic_features:
                colors.append(feature_color_map["electronic"])
            elif feature in intercalation_features:
                colors.append(feature_color_map["intercalation"])
        fig = plt.figure(dpi = 200, figsize=(6,6))
        ax = fig.gca()
        ax.bar(c, b, color=colors)
        ax.set_xticks(c)
        ax.set_xticklabels(a_modified, rotation=90, ha='center')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)
        ax.tick_params(bottom = True, top = True, left = True, right = True)
        ax.tick_params(axis = "x", direction = "in")
        ax.tick_params(axis = "y", direction = "in")
        plt.xlabel("Feature")
        plt.ylabel(f"Frequency in top models (Train CV score > {cv_score_cutoff})")
        plt.savefig(f"feature_counts.png", bbox_inches='tight')
        return fig

    def test(self, 
             X_test: Optional[DataFrame]=None,
             cv_score_cutoff: float=0.5,
             excluded_features: Optional[List[str]]=None
             ) -> DataFrame:
        """Tests the trained models. The test is done using the top models based on the cross-validation score.

        Args:
            X_test (Optional[DataFrame], optional): Input material features. If None, `X_test.pkl` should be present in the working directory. Defaults to None.
            cv_score_cutoff (float, optional): CV score cutoff to determine top models. Defaults to 0.5.
            excluded_features (Optional[List[str]], optional): All models which contain the excluded features are not used to test. Defaults to None.

        Returns:
            DataFrame: Average prediction from the top models for all materials in `X_test`.
        """
        self.process_train_results()
        if X_test is None:
            try:
                X_test = pd.read_pickle("X_test.pkl")
            except:
                raise FileNotFoundError(
                f"`X_test` is not provided and `X_test.pkl` does not exist.\n"
            )
        materials = X_test["material"]
        metadata = self.metadata
        top_models_info = metadata[metadata["cv_score"] > cv_score_cutoff]
        predictions = []
        for _, row in top_models_info.iterrows():
            job_id = row['job_id']
            features = row['features']
            if excluded_features and any(feature in excluded_features for feature in features):
                continue
            try:
                X_test_subset = X_test[list(features)]
            except KeyError:
                logging.info(f"Features {features} not found in `X_test`. Skipping test using model {job_id}.")
                continue
            try:
                pipeline = self.models[job_id]
                prediction = pipeline.predict(X_test_subset.values)
                predictions.append(prediction)
            except Exception as e:
                logging.error(f"Skipping test using model {job_id} due to the following error.\n{e}")
        avg_predictions = np.mean(predictions, axis=0)
        avg_predictions = pd.DataFrame({
            "material": materials,
            "capacity_ratio": avg_predictions
        })
        return avg_predictions