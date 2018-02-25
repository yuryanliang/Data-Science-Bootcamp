import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, LarsCV, LassoLarsCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import xgboost as xgb
#import h2o
#from h2o.estimators.glm import H2OGeneralizedLinearEstimator
#from h2o.estimators.random_forest import H2ORandomForestEstimator
#from h2o.estimators.gbm import H2OGradientBoostingEstimator
#h2o.init()

user = "Aritra"

#####################################################################
# Reading, Cleaning & Transforming Training Data                    #
# Splitting Training Dataset into Training and Testing Set          #
#####################################################################

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

train_data = pd.read_csv("../data/train.csv", index_col=0)
data = train_data
data = data.drop("SalePrice", axis=1)
test_data = pd.read_csv("../data/test.csv", index_col=0)
data = data.append(test_data)
cols = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 
        'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 
        'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 
        'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 
        'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 
        'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 
        'MoSold', 'YrSold', 'SaleType', 'SaleCondition']
cont_cols = np.setdiff1d(data.columns, cols)
for c in cols:
    tmp = np.sort(list(set(data[c])))
    for t in tmp:
        col_name = c + "_" + str(t)
        data[col_name] = np.zeros(data.shape[0])
        inds = []
        data.ix[data[c]==t, col_name] = 1.0
    data = data.drop(c, axis=1)
for c in cont_cols:
    data.ix[np.isnan(data[c]), c] = np.mean(data[c])
    data.ix[:, c] = normalize(data[c])
test_data = data.ix[test_data.index]
test_data.to_csv("../data/cleaned_test_data.csv")
data = data.ix[train_data.index]
data = pd.concat([data, train_data["SalePrice"]], axis=1)
data.to_csv("../data/cleaned_train_data.csv")

#####################################################################
# Reading the Cleaned Training & Testing Data                       #
#####################################################################

train_data = pd.read_csv("../data/cleaned_train_data.csv")
h2o_train_data = h2o.import_file("../data/cleaned_train_data.csv")
test_data = pd.read_csv("../data/cleaned_test_data.csv")
h2o_test_data = h2o.import_file("../data/cleaned_test_data.csv")
x_columns = test_data.columns[1:677]
y_column = "SalePrice"
x_train_values, y_train_values = train_data[x_columns].values, train_data[y_column].values
xgb_train_values = train_data[np.append(y_column, x_columns)].values
x_test_values = test_data[x_columns].values

def generate_submission_file(predictions, Id, filename):
    results = pd.DataFrame()
    results["Id"] = Id
    results["SalePrice"] = predictions
    results.to_csv(filename, index=False)
    
#####################################################################
# (High Dimensional) Linear Regression                              #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

lasso_model = LassoCV()
lasso_model.fit(x_train_values, y_train_values)
lasso_model_predictions = lasso_model.predict(x_test_values)
generate_submission_file(lasso_model_predictions, test_data["Id"], "../results/" + user + "_LassoCV.csv")

lars_model = LarsCV()
lars_model.fit(x_train_values, y_train_values)
lars_model_predictions = lars_model.predict(x_test_values)
generate_submission_file(lars_model_predictions, test_data["Id"], "../results/" + user + "_LarsCV.csv")

lassolars_model = LassoLarsCV()
lassolars_model.fit(x_train_values, y_train_values)
lassolars_model_predictions = lassolars_model.predict(x_test_values)
generate_submission_file(lassolars_model_predictions, test_data["Id"], "../results/" + user + "_LassoLarsCV.csv")

en_model = ElasticNetCV()
en_model.fit(x_train_values, y_train_values)
en_model_predictions = en_model.predict(x_test_values)
generate_submission_file(en_model_predictions, test_data["Id"], "../results/" + user + "_ElasticNetCV.csv")

#####################################################################
## XGBoost                                                         ##
#####################################################################

#####################################################################
## H2O                                                             ##
#####################################################################

#####################################################################
# Nearest Neighbors                                                 #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

knn_model = KNeighborsRegressor()
knn_model.fit(x_train_values, y_train_values)
knn_model_predictions = knn_model.predict(x_test_values)
generate_submission_file(knn_model_predictions, test_data["Id"], "../results/" + user + "_KNN.csv")

param_list = {"n_neighbors": [2, 4, 6]}
knn_gridsearch = GridSearchCV(KNeighborsRegressor(), param_list)
knn_gridsearch.fit(x_train_values, y_train_values)
knn_best_model_predictions = knn_gridsearch.best_estimator_.predict(x_test_values)
generate_submission_file(knn_best_model_predictions, test_data["Id"], "../results/" + user + "_KNN_GridSearchCV.csv")

#####################################################################
# Decision Trees                                                    #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

dt_model = DecisionTreeRegressor()
dt_model.fit(x_train_values, y_train_values)
dt_model_predictions = dt_model.predict(x_test_values)
generate_submission_file(dt_model_predictions, test_data["Id"], "../results/" + user + "_Decision_Tree.csv")

param_list = {"max_depth": np.linspace(10, len(x_columns), 100, dtype=np.int64)}
dt_gridsearch = GridSearchCV(DecisionTreeRegressor(), param_list)
dt_gridsearch.fit(x_train_values, y_train_values)
dt_best_model_predictions = dt_gridsearch.best_estimator_.predict(x_test_values)
generate_submission_file(dt_best_model_predictions, test_data["Id"], "../results/" + user + "_Decision_Tree_GridSearchCV.csv")

#####################################################################
## XGBoost                                                         ##
#####################################################################

#####################################################################
# Random Forests                                                    #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(x_train_values, y_train_values)
rf_model_predictions = rf_model.predict(x_test_values)
generate_submission_file(rf_model_predictions, test_data["Id"], "../results/" + user + "_Random_Forests.csv")

param_list = {"n_estimators": np.linspace(100, 500, 5, dtype=np.int64)}
rf_gridsearch = GridSearchCV(RandomForestRegressor(), param_list)
rf_gridsearch.fit(x_train_values, y_train_values)
rf_best_model_predictions = rf_gridsearch.best_estimator_.predict(x_test_values)
generate_submission_file(rf_best_model_predictions, test_data["Id"], "../results/" + user + "_Random_Forests_GridSearchCV.csv")

#####################################################################
## XGBoost                                                         ##
#####################################################################

#####################################################################
## H2O                                                             ##
#####################################################################

#h2o_rf = H2ORandomForestEstimator(ntrees=200, max_depth=6)
#h2o_rf.train(x=list(x_columns), y=y_column, training_frame=h2o_train_data)
#h2o_rf_predictions = h2o_rf.predict(h2o_test_data)
#generate_submission_file(h2o_rf_predictions, test_data["Id"], "../results/" + user + "_H2O_RF.csv")

#####################################################################
# Ada Boost                                                         #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

ada_model = AdaBoostRegressor()
ada_model.fit(x_train_values, y_train_values)
ada_model_predictions = ada_model.predict(x_test_values)
generate_submission_file(ada_model_predictions, test_data["Id"], "../results/" + user + "_Ada_Boost.csv")

param_grid = {"n_estimators": np.linspace(10, 100, 10, dtype=np.int64)}
ab_gridsearch = GridSearchCV(AdaBoostRegressor(), param_grid)
ab_gridsearch.fit(x_train_values, y_train_values)
ab_best_model_predictions = ab_gridsearch.best_estimator_.predict(x_test_values)
generate_submission_file(ab_best_model_predictions, test_data["Id"], "../results/" + user + "_Ada_Boost_GridSearchCV.csv")

#####################################################################
# Gradient Boosting Machines                                        #
#####################################################################

#####################################################################
## Scikit Learn                                                    ##
#####################################################################

gbm_model = GradientBoostingRegressor()
gbm_model.fit(x_train_values, y_train_values)
gbm_model_predictions = gbm_model.predict(x_test_values)
generate_submission_file(gbm_model_predictions, test_data["Id"], "../results/" + user + "_Gradient_Boosted_Machines.csv")

param_grid = {"n_estimators": np.linspace(10, 150, 15, dtype=np.int64)}
gbm_gridsearch = GridSearchCV(GradientBoostingRegressor(), param_grid)
gbm_gridsearch.fit(x_train_values, y_train_values)
gbm_best_model_predictions = gbm_gridsearch.best_estimator_.predict(x_test_values)
generate_submission_file(gbm_best_model_predictions, test_data["Id"], "../results/" + user + "_Gradient_Boosted_Machines_GridSearchCV.csv")

#####################################################################
## XGBoost                                                         ##
#####################################################################

#####################################################################
### Weak Learner is a Tree                                        ###
#####################################################################

xgb_model = xgb.XGBRegressor()
xgb_model.fit(x_train_values, y_train_values)
xgb_model_predictions = xgb_model.predict(x_test_values)
generate_submission_file(xgb_model_predictions, test_data["Id"], "../results/" + user + "_XGBoost_Basic.csv")

param_grid = {"max_depth": [2,4,6],
              "n_estimators": np.linspace(100, 500, 5, dtype=np.int64)}
xgb_grid_search = GridSearchCV(xgb.XGBRegressor(objective="reg:linear"), param_grid)
xgb_grid_search.fit(x_train_values, y_train_values)
xgb_model_predictions = xgb_grid_search.predict(x_test_values)
generate_submission_file(xgb_model_predictions, test_data["Id"], "../results/" + user + "_XGBoost_GridSearchCV.csv")

#####################################################################
## H2O                                                             ##
#####################################################################

#h2o_gbm = H2OGradientBoostingEstimator(ntrees=100, max_depth=6)
#h2o_gbm.train(x=list(x_columns), y=y_column, training_frame=h2o_train_data)
#h2o_gbm_predictions = h2o_gbm.predict(h2o_test_data)
#generate_submission_file(h2o_gbm_predictions, test_data["Id"], "../results/" + user + "_H2O_GBM.csv")

#####################################################################
# Stacked Ensembles                                                 #
#####################################################################

#####################################################################
## H2O                                                             ##
#####################################################################
