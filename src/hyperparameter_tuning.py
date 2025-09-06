import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.datasets import load_breast_cancer



#loading the data
data = load_breast_cancer()
X = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target,name="target")

# splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
rf = RandomForestClassifier(random_state=42)


params_grid = {
    'n_estimators':[10,25,50,75,100],
    'max_depth': [None,10,20,30]
}

grid_search = GridSearchCV(estimator=rf,param_grid=params_grid)

# starting an mlflow run
mlflow.set_experiment("breast_cancer_hypertuning")
with mlflow.start_run() as parent:
    grid_search.fit(X_train,y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # Log test data
    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    
    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")
    
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Bhanuprasad', "Project": "Breast_cancer_hypertuning"})


    print(best_params)
    print(best_score)

