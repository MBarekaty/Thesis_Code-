### Importing processing libraries


```python
import os
import rasterio as rio
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt
from osgeo import gdal
from osgeo import gdal_array
from osgeo import osr
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
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
use_vegetation_indices =True
use_hyperparameter_tuning =False
use_oversampling=False
```


```python
train_features = pd.read_csv('Train_split_70_resampled.csv')
print('We have {} train data with {} variables'.format(*train_features.shape))
test_features = pd.read_csv('Test_split_30_resampled.csv')
print('We have {} test data with {} variables'.format(*test_features.shape))

```

#### Training Process
<blockquote><b>RF</b><br>Random forest processing steps it starts with predicting and finding the accuracy and trying different options such as <b>using oversampling</b>, using <b>hyperparameter tuning</b> and <b>Vegetation indices</b> the result of each option has been provided below</blockquote>

<b>Extracting value for test and train dataset for randomforest assessment</b><br>
<b>Training process without considering different indices</b>


```python
if  use_vegetation_indices == True: 
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

### Using Oversampling


```python
if use_oversampling == True:
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(X_train, y_train)
print('Resampled train dataset shape %s' % Counter(y_train))

```

## Training  model 


```python
if use_hyperparameter_tuning == True:
    rf = RandomForestClassifier(random_state=42)
    parameters={"n_estimators":[100, 200],
              "max_depth":[3, 5, 6, 9, 15], 'min_samples_split': [11, 15, 19], 
              'min_samples_leaf': [3,5,7], "bootstrap":[True,False],"max_features":['auto']}
    grid_rf = GridSearchCV(rf, param_grid = parameters, cv = 3, n_jobs = -2)
    grid_rf.fit(X_train, y_train)
    scores_df = pd.DataFrame(grid_rf.cv_results_) #scores_df = pd.DataFrame(grid_rf.cv_results_)
    best_param=grid_rf.best_params_ #best=grid_rf.best_params_
    print("Best paramters :", grid_rf.best_params_)
    print("Best accuracy scores for train data:", grid_rf.best_score_)
    pred = grid_rf.predict(X_test)# Prediction
    Accuracy_hyperparameter =accuracy_score(pred,y_test)
    print("Accuracy score based on test data with Hyperparameter tuning",round(metrics.accuracy_score(pred,y_test)*100,2), '%.')
    print(classification_report(y_test,pred))
    fig, ax = plt.subplots(figsize=(10, 10))
    print("Confusion Metrix:\n",metrics.confusion_matrix(pred,y_test))
    cm = metrics.confusion_matrix(pred,y_test)
    plot_confusion_matrix(grid_rf, X_test, y_test ,  ax=ax,cmap=plt.cm.YlGnBu)
    plt.title('Confusion matrix RF hyperparameter model')
    plt.savefig("figure2.png") 
    plt.show()
    #normalize='true'

else:
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                                max_features='auto',random_state=42,max_samples=None)
    #learning
    rf.fit(X_train,y_train)
    #Prediction
    prediction=rf.predict(X_test)
    #evaluation(Accuracy)
    #Accuracy_base_model= accuracy_score(prediction,y_test)
    Accuracy_base_model= accuracy_score(prediction,y_test)
    print("Accuracy score based on test data without Hyperparameter tuning",round(metrics.accuracy_score(prediction,y_test)*100,2), '%.')
    #evaluation(Confusion Metrix)
    print(classification_report(y_test,prediction,zero_division=0))
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_confusion_matrix(rf, X_test, y_test ,  ax=ax,cmap=plt.cm.YlGnBu)
    print("Confusion Metrix:\n",metrics.confusion_matrix(prediction,y_test))
    cm = metrics.confusion_matrix(prediction,y_test)
    plt.title('Confusion matrix of Random forest model')
    plt.savefig("figure.png") 
    plt.show()
    
```

### Resampled orthophoto UAV by the model with higher accuracy 


```python
# Loading the resampled (121 pixels) raster data
raster_filename = "Ortho_compelete_resampled.tif"
DSM="dsm_re.tif"
GLI="GLI_resampled.tif"
VARI="Vari_resampled.tif"
```


```python
dataset = rio.open(raster_filename)
dsm=rio.open(DSM)
vari=rio.open(VARI)
gli=rio.open(GLI)
```


```python
#Reading the raster values
band1 = dataset.read(1)
band2=dataset.read(2)
band3=dataset.read(3)
dsm2 = dsm.read(1)
vari2 = vari.read(1)
gli2=gli.read(1)
```


```python
## Converting 2D numpy array to 1d array
```


```python
## RGB values 
one=band1.flatten() 
two=band2.flatten() 
three=band3.flatten()
one_data= pd.DataFrame(one)
two_data= pd.DataFrame(two)
th_data= pd.DataFrame(three)
dataframe_rgb=pd.concat([one_data,two_data,th_data], axis=1)
dataframe_rgb.columns =['red', 'blue', 'green']
```


```python
## DSM & GLI & VARI
dsm2_r = dsm2.flatten()   
vari2_r = vari2.flatten()
gli2_r = gli2.flatten()
df1=pd.DataFrame(dsm2_r)
df2=pd.DataFrame(vari2_r)
df3=pd.DataFrame(gli2_r)
indices=pd.concat([df1,df3,df2], axis=1)
indices.columns =['DSM','GLI', 'VARI']
```

#### Making the dataframe of resampled data
Making the simliar dataframe to the main dataframe for prediction of labels of each pixels


```python
dataframe2=pd.concat([dataframe_rgb,indices], axis=1)
```


```python
dataframe2.loc[(dataframe2.red== 0.0) & (dataframe2.blue== 0.0) & (dataframe2.green== 0.0),'DSM']=0.0
dataframe2.loc[(dataframe2.red== 0.0) & (dataframe2.blue== 0.0) & (dataframe2.green== 0.0) ,'GLI']=0.0
dataframe2.loc[(dataframe2.red== 0.0) & (dataframe2.blue== 0.0) & (dataframe2.green== 0.0),'VARI']=0.0
```

#### Prediction of  each pixels labels


```python
prediction1=rf.predict(dataframe2)
```


```python
labels1 = pd.Series(prediction1)
```


```python
dataframe1=pd.concat([dataframe2,labels1], axis=1)
dataframe1.rename(columns={0: 'labels'}, inplace=True)
```


```python
x=dataframe1['labels'].value_counts()
```


```python
dataframe1.loc[(dataframe1.red== 0.0) & (dataframe1.blue== 0.0) & (dataframe1.green== 0.0),'labels']=0.0
```


```python
dataframe1['labels'].value_counts()
```


```python
labels= dataframe1['labels']
```

### Visualization of final classified map 


```python
import matplotlib.colors
raster = np.ndarray(shape=(10,20), dtype=np.float32)
raster=labels.values.reshape(4575,3786)
```


```python
fig,ax=plt.subplots(figsize=(10,10))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ['w','dodgerblue','forestgreen','darkkhaki','olivedrab','khaki','seagreen','yellowgreen','brown'])
#['w','dodgerblue','forestgreen','darkkhaki','olivedrab','y','seagreen','yellowgreen','brown'])
plt.imshow(raster,cmap=cmap)
c=plt.colorbar(ticks=range(9), label='landcover classes ')
c.ax.set_yticklabels(['Background','1: Water', '2:Pinus', ' 3: Betula','4:Rhynchospora','5:Phragmites','6: Eriphorum','7:Carex','8:Barepeat'])
plt.title('The reclassified map of study area with RF classifier')
plt.savefig("test.png")
plt.show()
```


```python
array=raster
xmin,ymin,xmax,ymax = [521300.8197345125372522,6494659.9188134595751762,
                       522491.9864475125796162,6496099.11398845911] #meters (obtained with a conversion)
nrows,ncols = np.shape(array)
xres = (xmax-xmin)/float(ncols)
yres = (ymax-ymin)/float(nrows)
geotransform=(xmin,xres,0,ymax,0, -yres) 
output_raster = gdal.GetDriverByName('GTiff').Create('final_test.tif',ncols, nrows, 1 ,gdal.GDT_Float32)
#writting output raster
output_raster.GetRasterBand(1).WriteArray( array ) 
output_raster.SetGeoTransform(geotransform)
srs = osr.SpatialReference()
srs.ImportFromEPSG(3301) 
output_raster.SetProjection(srs.ExportToWkt())
output_raster = None

```


```python
#plt.savefig('finalmap.png')
plt.savefig("finalmap.png")
```

### Counting the number of True and False predicted labels


```python
df = test_features.iloc[:,0:2]
```


```python
df1 = pd.DataFrame(data=prediction)
df1.rename(columns={0: 'PL'}, inplace=True)
```


```python
df_col = pd.concat([df, df1],axis = 1)
#path=r"C:\Thesis_data\Data\working"
#Half = os.path.join(path,'Half.csv')
#df_col.to_csv(Half, index=False)
```


```python
df_compare = np.where(df_col['PL'] == df_col['Class'], 'True', 'False')
df_compare=pd.DataFrame(data=df_compare)
```


```python
df3 = pd.concat([df_compare, df_col],axis = 1)
df3.rename(columns={0: 'lables'}, inplace=True)
df3.head(5)
```


```python
print(df3['PL'].value_counts())
```


```python
y=df3['lables'].value_counts()
```


```python
count_class = pd.DataFrame(df3.groupby('lables').size())
ax = df3.lables.value_counts()
.plot(kind='barh', title="Predicted lables of test data ",color='red')
ax.set_xlabel("Predicted lables")
ax.set_ylabel("Count")

```


```python

```
