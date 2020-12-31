### Importing processing libraries


```python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from rasterio.plot import show
from collections import Counter
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
```


```python
#Workspace path
os.chdir(r"C:\Thesis_data\Data\working")
```

### Importing setup assessment and initial data 
##### Processing Parameters  
Changing Parameters from TRUE to FALSE


```python
use_vegetation_indices = True
use_MinMaxScaler= True
use_hyperparameter_tuning = True
use_oversampling =False
```


```python
train_features = pd.read_csv('Train_split_70_p121.CSV')
print('We have {} train data with {} variables'.format(*train_features.shape))
test_features = pd.read_csv('Test_split_30_p121.CSV')
print('We have {} test data with {} variables'.format(*test_features.shape))
```

### Training process
<blockquote><b>SVM : </b><br>Support Vector Machines steps it starts with predicting and finding the accuracy and trying different options such as <b>using oversampling</b>, using <b>hyperparameter tuning</b> and <b>Vegetation indices</b> the result of each option has been provided below</blockquote>

<b>Extracting value for test and train dataset for SVM assessment</b><br>
<b>Training process without considering different indices</b>


```python
if use_vegetation_indices == True:
    # included vegetation indices
    X_train = train_features.iloc[:,2:]
    y_train = train_features.iloc[:,0]
    X_test = test_features.iloc[:,2:]
    y_test = test_features.iloc[:,0]

else:
    # Included DSM
    X_train = train_features.iloc[:,2:6]
    y_train = train_features.iloc[:,0]
    X_test = test_features.iloc[:,2:6]
    y_test = test_features.iloc[:,0]
```


```python
print('Training Features Shape:', X_train.shape)
print('Training Labels Shape:', y_train.shape)
print('Testing Features Shape:', X_test.shape)
print('Testing Labels Shape:', y_test.shape)
```

### Using MinMaxScaler


```python
if use_MinMaxScaler==True:
    trans = MinMaxScaler()
    X_train = trans.fit_transform(X_train)
    X_test = trans.transform(X_test)
```

### Using Oversampling 


```python
if use_oversampling == True:
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
    print('Resampled train dataset shape %s' % Counter(y_train))
```

## Training model


```python

if use_hyperparameter_tuning == True:
    svm = SVC(probability = True)
    parameters ={"kernel":['rbf'], 'C': [1,10],'gamma' : [0.001,0.1], 'degree':[1, 2]}
    grid_svm = GridSearchCV(svm, param_grid = parameters, cv = 3, n_jobs = -1)
    grid_svm.fit(X_train, y_train)
    scores_df = pd.DataFrame(grid_svm.cv_results_)
    print("Best paramters:", grid_svm.best_params_)
    print("Best accuracy scores:", grid_svm.best_score_)
    pred = grid_svm.predict(X_test)
    print("Accuracy for SVM hyperparameter model :",round(metrics.accuracy_score(pred,y_test)*100,2), '%.')
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred))
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(grid_svm, X_test, y_test ,  ax=ax,cmap=plt.cm.YlGnBu)
    plt.title('Confusion matrix SVM hyperparameter model')
    plt.savefig("figure3.png") 
    plt.show()
else:
    
    from sklearn import svm
    model=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='scale', verbose=False,random_state=None)
    #learning
    model.fit(X_train,y_train)
    #Prediction
    prediction=model.predict(X_test)
    #evaluation(Accuracy)
    print("Accuracy for SVM base model :",round(metrics.accuracy_score(prediction,y_test)*100,2), '%.')
    #evaluation(Confusion Metrix)
    cm=metrics.confusion_matrix(prediction,y_test)
    print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
    print(classification_report(y_test,prediction))
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(model, X_test, y_test ,  ax=ax,cmap=plt.cm.YlGnBu)
    plt.title('Confusion matrix SVM base model')
    plt.savefig("figure.png") 
    plt.show()
```

### Counting the number of True and False predicted labels


```python
df = test_features.iloc[:,0:2]
```


```python
df1 = pd.DataFrame(data=prediction)
df1.rename(columns={0: 'Predicted Labels'}, inplace=True)
```


```python
df_col = pd.concat([df, df1],axis = 1)
##path=r"C:\Thesis_data\Data\working"
#Halfp11 = os.path.join(path,'Halfp11.csv')
#df_col.to_csv(Halfp11, index=False)
```


```python
df_compare = np.where(df_col['Predicted Labels'] == df_col['Class'], 'True', 'False')
df_compare=pd.DataFrame(data=df_compare)
```


```python
df3 = pd.concat([df_compare, df_col],axis = 1)
df3.rename(columns={0: 'Compared Labels'}, inplace=True)
```


```python
print(df3['Compared Labels'].value_counts())
```


```python

```
