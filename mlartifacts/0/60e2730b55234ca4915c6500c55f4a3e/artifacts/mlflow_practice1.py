import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt 
from sklearn import metrics


#loading the data
wine = load_wine()
X = wine.data
y = wine.target

# splitting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)

# parameter
max_depth = 7
n_estimators = 10

# starting an mlflow run
mlflow.set_tracking_uri('http://localhost:5000')
with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators)
    rf.fit(X_train,y_train)
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
# logging metrics and parameter
    mlflow.log_metric('accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

# plotting a confusion matrics
    confusion_matric = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matric, display_labels = [0, 1])
    plt.savefig("confusion-matrix.png")

    # logging artifacts like confusion matrix and python file and model as well 
    mlflow.log_artifact('confusion-matrix.png')
    mlflow.log_artifact(__file__)

    # tags
    mlflow.set_tags({"Author": 'Bhanuprasad', "Project": "Wine Classification"})

    # Log the model
    mlflow.sklearn.log_model(rf, "Random-Forest-Model")

print(accuracy)

