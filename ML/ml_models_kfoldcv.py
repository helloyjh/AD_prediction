from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Lasso, RidgeClassifier, SGDClassifier, Lars, LassoLars, ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.exceptions import ConvergenceWarning
#from sklearn.utils.testing import ignore_warnings
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np


import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
#@ignore_warnings(category=ConvergenceWarning)
class ML_Models():
    def __init__(self, X, Y, cv=5, scoring=None, n_cores=10):
        self.train_X, self.train_Y = X, Y
        self.cv = cv
        self.scoring = scoring
        self.auc = {}
        self.n_cores = n_cores

        self.train_Y = self.train_Y.ravel()
    def _auc_(self):
        return self.auc

    def ml_xgb(self, params=None):
        model = XGBClassifier(use_label_encoder=False)


        if not params:
            params = {
                'max_depth': [1, 3, 5],
                'subsample': [0.5, 1.0],
                'learning_rate': [0.01, 0.05, 0.1,  0.2],
                'n_estimators': [100, 500, 1000, 2000]
            }


        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("extreme gradient boost")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")


        return model_grid

    def ml_linear_xgb(self, params=None):
        model = XGBRegressor(eval_metric=mean_absolute_error,n_jobs=self.n_cores)

        if not params:
            params = {
                'max_depth': [3, 5, 10, None],
                'gamma' : [0, 0.1, 0.2],
                'subsample': [0.5, 1.0],
                'learning_rate': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.2],
                'min_child_weight' : [1,2,5],
                'n_estimators': [10, 100, 500, 1000, 2000],
                'lambda' : [0.1, 1, 2]
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0,
                                  n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)

        # best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("extreme gradient boost")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid
        #return model


    def ml_svm(self, params=None):
        model = SVC()

        # if not params:
        #     params = {
        #         'C': [0.001, 0.01, 0.1, 0.5, 1, 10, 100],
        #         'gamma': [0.001, 0.01, 0.1, 1, 2, 5],
        #         'kernel': ['poly','rbf','sigmoid'],
        #         'degree': [2, 3, 4],
        #         'class_weight': ['balanced'],
        #         'probability': [True]
        #     }

        params = {
            'C': [0.001, 0.01, 0.005, 0.1, 0.5, 1, 5, 10, 20],
            'gamma': [0.001, 0.005, 0.01, 0.005, 0.1, 0.5, 1],
            'class_weight': ['balanced'],
            'probability': [True]
        }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=-self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("svm")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid

    def ml_linear_svm(self, params=None):
        model = SVR()


        params = {
            'kernel' : ['rbf', 'linear','sigmoid'],
            'C': [0.01, 0.1, 0.5, 1, 2, 5, 10],
        }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=-self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("svm")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid

    def ml_logistic(self, params=None):
        model = LogisticRegression()

        # if not params:
        #     params = {
        #         'penalty': ['l1', 'l2', 'elasticnet'],
        #         'solver': ['saga'],
        #         'C': [0.001, 0.01, 0.1, 1, 10, 100],
        #         'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        #         'max_iter': [100, 500],
        #         'class_weight': ['balanced']
        #     }

        if not params:
            params = {
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['saga'],
                'C': [0.001, 0.01, 0.02, 0.1, 0.5, 1, 10, 100],
                'l1_ratio': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'class_weight': ['balanced']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)


        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("logistic")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid
        #return model

    def ml_linear_rf(self, params=None):
        model = RandomForestRegressor()

        if not params:
            params = {
                'max_depth': [3, 5, 10, 20, 50, None],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features' : ['auto','sqrt'],
                'n_estimators': [100, 300, 500, 1000],
                'criterion': ['squared_error', 'absolute_error']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("random forest")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid
        #return model

    def ml_rf(self, params=None):
        model = RandomForestClassifier()

        if not params:
            params = {
                'max_depth': [3, 5, 10, 20, 50],
                'min_samples_leaf': [1, 2, 4, 5, 10, 20],
                'n_estimators': [100, 300, 500, 1000],
                'class_weight': ['balanced', 'balanced_subsample']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("random forest")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid

    def ml_extraTrees(self, params=None):
        model = ExtraTreesClassifier()

        # if not params:
        #     params = {
        #         'max_depth': [None, 3, 5, 7, 9],
        #         'n_estimators': [10, 50, 100, 300, 500],
        #         'class_weight': ['balanced', 'balanced_subsample']
        #     }

        if not params:
            params = {
                'max_depth': [None, 3, 10, 50, 100, 200, 500],
                'n_estimators': [10, 50, 100, 300, 500, 1000],
                'max_features' : ['auto', 'sqrt'],
                'class_weight': ['balanced', 'balanced_subsample']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)


        #best_model = model_grid.best_estimator_

        # print("##############################################")
        # print("extraTrees")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid
        #return model

    def ml_gradientboost(self, params=None):
        model = GradientBoostingClassifier()

        # if not params:
        #     params = {
        #         'loss': ['deviance', 'exponential'],
        #         'n_estimators': [100, 300, 500, 1000],
        #         'learning_rate': [0.01, 0.05, 0.1, 0.5],
        #         'max_depth': [10, 50, 100, 200, 500, None]
        #     }

        if not params:
            params = {
                'n_estimators': [50, 100, 300, 500, 1000, 2000],
                'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.5],
                'max_depth': [1, 2, 3, None]
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)


        #best_model = model_grid.best_estimator_
        # print("##############################################")
        # print("gradient boost")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")


        return model_grid
        #return model

    def ml_lightgbm(self, params=None):
        model = LGBMClassifier()

        # if not params:
        #     params = {
        #         'min_child_weight': [0.5, 1, 2, 5],
        #         'colsample_bytree': [0.6, 0.8, 1.0],
        #         'subsample': [0.6, 0.8, 1.0],
        #         'learning_rate': [0.05, 0.1],
        #         'n_estimators': [100, 300],
        #         'reg_alpha': [0.0, 1.0, 2.0, 5.0],
        #         'reg_lambda': [0.0, 1.0, 2.0, 5.0],
        #         'class_weight': ['balanced']
        #     }

        if not params:
            params = {
                'subsample': [0.5, 1.0],
                'learning_rate': [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                'max_depth' : [1,2,3],
                'n_estimators': [100, 500, 1000],
                'class_weight': ['balanced']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_
        # print("##############################################")
        # print("light gbm")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid

    def ml_linear_lightgbm(self, params=None):
        model = LGBMRegressor()

        if not params:
            params = {
                'subsample': [0.5, 1.0],
                'learning_rate': [0.0001, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
                'max_depth' : [1,3,5,10,20,-1],
                'objective' : ['regression','regression_l1'],
                'n_estimators': [10, 50, 100, 500, 1000, 2000],
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0, n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)
        #model.fit(self.train_X, self.train_Y)

        #best_model = model_grid.best_estimator_
        # print("##############################################")
        # print("light gbm")
        # print("Best Score : {}".format(model_grid.best_score_))
        # print("Best Params : {}".format(model_grid.best_params_))
        # print("##############################################")

        return model_grid

    def ml_lasso(self, params=None):
        model = Lasso()

        if not params:
            params = {
               'alpha': [0.001, 0.01, 0.02, 0.1, 0.5, 1, 10, 100]
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0,
                                  n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)


        return model_grid

    def ml_ridge(self, params=None):
        model = Ridge()

        if not params:
            params = {
                'alpha': [0.001, 0.01, 0.02, 0.1, 0.5, 1, 10, 100],
                'solver' : ['auto']
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0,
                                  n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)


        return model_grid

    def ml_enet(self, params=None):
        model = ElasticNet()

        if not params:
            params = {
                'alpha': [0.001, 0.01, 0.02, 0.1, 0.5, 1, 10, 100],
                'l1_ratio': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            }

        model_grid = GridSearchCV(model, param_grid=params, scoring=self.scoring, cv=self.cv, verbose=0,
                                  n_jobs=self.n_cores)
        model_grid.fit(self.train_X, self.train_Y)


        return model_grid
#
# import os
# from glob import glob
# for x in a:
#     n=x.split('/')[-1]
#     s=n.split('-')[-1]
#     if 'S-' in n:
#         os.system('mv ' + n + ' ' + 'S_' + s)
#
