from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow

import dagshub
dagshub.init(repo_owner='Arihant987', repo_name='ML_Flow', mlflow=True)

mlflow.set_tracking_uri("https://dagshub.com/Arihant987/ML_Flow.mlflow")

data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='target')

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)

rf=RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}
# cv is cross validation, n_jobs is number of jobs to run in parallel
# Without mlflow
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs=-1,verbose=2)
'''
grid_search.fit(x_train,y_train)

best_params=grid_search.best_params_
best_score=grid_search.best_score_

print(f"Best params: {best_params}")
print(f"Best score: {best_score}")
'''
# With mlflow
mlflow.set_experiment("exp-cancer")

with mlflow.start_run() as parent:
    grid_search.fit(x_train,y_train)

    for child in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_params(grid_search.cv_results_['params'][child])
            mlflow.log_metric("accuracy",grid_search.cv_results_['mean_test_score'][child])

    best_params=grid_search.best_params_
    best_score=grid_search.best_score_

    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy",best_score)

    # log train data and test data
    train_df=x_train.copy()
    train_df['target']=y_train

    # mlflow datatype
    train_df=mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df,"train_data")

    test_df=x_test.copy()
    test_df['target']=y_test

    test_df=mlflow.data.from_pandas(test_df)    
    mlflow.log_input(test_df,"test_data")

    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(grid_search.best_estimator_,"Random_Forest")

    mlflow.set_tags({'Author':'KekLmao','Project':"Breast Cancer Classification"})
    print(f"Best params: {best_params}")
    print(f"Best score: {best_score}")
