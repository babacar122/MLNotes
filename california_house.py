from pathlib import Path
import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from scipy.stats import randint
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, FunctionTransformer
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.plotting import scatter_matrix


class ClusterSimilarity(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, gamma=1.0, random_state=None):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.random_state = random_state

    def fit(self, X, y=None, sample_weight=None):
        self.kmeans_ = KMeans(self.n_clusters, random_state=self.random_state)
        self.kmeans_.fit(X, sample_weight=sample_weight)
        return self # always return self!
    
    def transform(self, X):
        return rbf_kernel(X, self.kmeans_.cluster_centers_, gamma=self.gamma)
        
    def get_feature_names_out(self, names=None):
        return [f"Cluster {i} similarity" for i in range(self.n_clusters)]



"""This function will be responsible for loading the data"""
def load_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open("tarball_path") as housing_tarball:
            housing_tarball.extractall(path="datasets")

    return pd.read_csv(Path("datasets/housing/housing.csv"))

"""This function gives unique hash values to the instances in the dataset. If the hash value of a given instance is below 20% of the maximum hash
values of the dataset, then it is added to the test set."""
def is_id_in_test_set(identifier, test_ratio):
    return crc32(np.int64(identifier)) < test_ratio * 2**32


"""A function to create a test set"""
def shuffle_and_split_data(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_tranformer, feature_names_in):
    return ["ratio"]

def ratio_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(column_ratio, feature_names_out=ratio_name), StandardScaler())


def main():
    housing = load_data()
    print("Print the data on housing")
    print(housing.head())
    print("Visualization of the training data...")
    #housing.hist()
    #plt.show()
    # since the housing dataset does not have an id column we need to add one
    housing_with_id = housing.reset_index()
    train_set, test_set = shuffle_and_split_data(housing_with_id, 0.2, "index")
    housing["income_cat"] = pd.cut(housing["median_income"], bins=[0., 1.5, 3.0, 4.5, 6, np.inf], labels=[1, 2, 3, 4, 5])
    # We are going to stratify the dataset
    strat_train_set, strat_test_set = train_test_split(housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)
    # Since the income category column is not needed anymore we can drop for performance sake
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)
    # Now we are going to visualize the geographical data
    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.2)
    plt.show()
    # And here is the housing prices data
    housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, s=housing["population"] / 100, label="population", c="median_house_value", cmap="jet", colorbar=True
    , legend=True, sharex=False, figsize=(10, 7))
    plt.show()
    # Let's now look for correlation between the house median value and all the other attributes
    print("Correlation between the different attributes and the median house value")
    attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12, 8))
    plt.show()
    # By observation we see a strong correlation between the median house value and the median income so let's focus on this graph
    housing.plot(kind="scatter", x="median_income", y="median_house_value", grid=True, alpha=0.1)
    plt.show()
    """Now we are going to prepare the data for our algorithm by first reverting to a clean dataset without the additional attributes we added to it"""
    housing = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    housing_num = housing.select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing_num) # This line store the median of all the numerical attributes in a statistics_ variable
    # print(imputer.statistics_) returns the median of all the numerical attributes
    X = imputer.transform(housing_num)
    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)
    # Now let's take care of the only categorical attribute "ocean_proximity"
    housing_cat = housing[["ocean_proximity"]]
    one_hot_encoder = OneHotEncoder()
    housing_cat_encoded = one_hot_encoder.fit_transform(housing_cat)
    # We are going to normalize all the numerical attributes so that our algorithm will not underperformed because of the difference in the scaling of the attributes
    #min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    #housing_min_max_scaled = min_max_scaler.fit_transform(housing_num)
    std_scaler = StandardScaler()
    housing_num_std_scaled = std_scaler.fit_transform(housing_num)
    # Now we are going to use a rbf scaler that deals with distribution with a heavy tail(features that have multiple modes)
    age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)
    # And we are going to apply a custom transformer to the population feature
    log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
    log_pop = log_transformer.transform(housing[["population"]])
    # we are going to use scikit-learn pipeline which is going to encapsulate all the transformation we need to do with our dataset
    # this one is for the numerical attributes
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("standardize", StandardScaler())])
    # Now we are going to use a transformer that is both capable of handling numerical and non numerical columns
    log_pipeline = make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log, feature_names_out="one-to-one"),StandardScaler())
    cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)
    default_num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    cat_pipeline = make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore"))
    preprocessing = ColumnTransformer([("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]), ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]), ("people_per_house", ratio_pipeline(), ["population", "households"]),
    ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population", "households", "median_income"]), ("geo", cluster_simil, ["latitude", "longitude"]), ("cat", cat_pipeline, make_column_selector(dtype_include=object)),], remainder=default_num_pipeline)
    # The next line use the all the preparation we have done and use it on the dataset
    housing_prepared = preprocessing.fit_transform(housing)
    # Now it's time to train models in our data. We are going to start with a simple linear regression
    lin_reg = make_pipeline(preprocessing, LinearRegression())
    lin_reg.fit(housing, housing_labels)
    
    print("Comparing the prediction and the data labels using Linear Regression")
    print("Predictions for the first five districts")
    housing_predictions = lin_reg.predict(housing)
    housing_predictions[:5].round(-2)
    print("Labels for the firt 5 districts")
    housing_labels.iloc[:5].values
    lin_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False) # Evaluating the error using RMSE
    print(lin_rmse)
    # This one does not produce good prediction so let's test a decision tree
    tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor(random_state=42))
    tree_reg.fit(housing, housing_labels)
    housing_predictions = tree_reg.predict(housing)
    tree_rmse = mean_squared_error(housing_labels, housing_predictions, squared=False)
    print("Evaluation from the decision tree regressor")
    print(tree_rmse)
    # Since this one produce allegedly 0.0 error(which means the models largely overfits the data) so we are going to use scikit-learn's k-fold cross validation for a beter evaluation
    tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    pd.Series(tree_rmses).describe()
    # This model is not satisfying either so let's try a Random Forest Regressor which is just a bunch of random tree put together
    forest_reg = make_pipeline(preprocessing, RandomForestRegressor(random_state=42))
    forest_rmses = -cross_val_score(forest_reg, housing, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
    print("Evaluation from the random forest regressor")
    pd.Series(forest_reg).describe()

    # Fine tuning our model

    # The next lines uses scikit-learn's grid search to explore the best set of hyperparameters to apply to our model
    full_pipeline = Pipeline([("preprocessing", preprocessing), ("random_forest", RandomForestRegressor(random_state=42))])
    param_grid = [{'preprocessing__geo__n_clusters': [5, 8, 10], 'random_forest__max_features': [4, 6, 8]}, 
    {'preprocessing__geo__n_clusters': [10, 15], 'random_forest__max_features': [6, 8, 10]}]
    grid_search = GridSearchCV(full_pipeline, param_grid, cv=3, scoring="neg_root_mean_squared_error")
    grid_search.fit(housing, housing_labels)
    print("Let's see the best hyperparameter combinations found by grid search cross validation:")
    print(grid_search.best_params_)
    param_distribs = {'preprocessing__geo__n_clusters': randint(low=3, high=50), 'random_forest__max_features': randint(low=2, high=20)}
    rnd_search = RandomizedSearchCV(full_pipeline, param_distributions=param_distribs, n_iter=10, cv=3, scoring='neg_root_mean_squared_error', random_state=42)
    rnd_search.fit(housing, housing_labels)

    # Now we are going to use our test set for a final evaluation
    final_model = rnd_search.best_estimator_
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()
    final_predictions = final_model.predict(X_test)
    final_rmse = mean_squared_error(y_test, final_predictions, squared=False)
    print(final_rmse)





if __name__ == "__main__":
    main()
