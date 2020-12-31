# Data Prepration
### Visualization  and Extracting of data value 


<i>Importing libraries for visualizing and extracting data</i>


```python
import os
import rasterio
import rasterio as rio
import seaborn as sns
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from subprocess import call
from rasterio.plot import show
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from IPython.display import Image
from rasterio.plot import plotting_extent
from sklearn.model_selection import train_test_split
```


```python
# Reading the file and workspace for data
os.chdir(r"C:\Thesis_data\Data\working")
path=r"C:\Thesis_data\Data\working"
path_to_output=r"C:\Thesis_data\Data\working"
```

#### Reading the initial data
 
 * Land cover classes shapefile
 * Orthophoto UAV image file of Maima area
 * DSM of Maima area
 * Vegetaion indices: Green leaf index(GLI) & Visible Atmospherically      Resistant Index (VARI)

#### Having different Train and Test split
<blockquote><b>Choosing different landcover classes for having different train and test split for finding the affect of spatial distribution of Train and Test split
Here, we have defined 3 different split between train and test randomly by selecting features  ranomly in Qgis.By choosing the right column for training the result will vary.</blockquote>


```python
use_different_split= False
use_25pixels= False
use_121pixels= True
```


```python
if use_different_split==True:
    shp_filename = "landcover2_Class.shp"
else:
    shp_filename = "Land_cover_classes1.shp"

    
raster_filename = "Ortho_Compelete.tif"
DSM="Maima_DSM_190724.tif"
GLI="GLI_CP.tif"
VARI="VARI_CP.tif"
```

<blockquote>Visualization of different spatial split of Test and Train data </blockquote>



```python
if use_different_split==True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    img = mpimg.imread('test_train_3rd1.png')
    img2=mpimg.imread('test_train_half1.png')
    img3=mpimg.imread('test_train_RANDOM2.png')
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(img2)
    ax.set_title('Train and Test data is equal,column: Half')
    ax = fig.add_subplot(1,3, 2)
    imgplot = plt.imshow(img3)
    ax.set_title('Randomly selection of Test and Train data,column:Random')
    ax = fig.add_subplot(1,3, 3)
    imgplot = plt.imshow(img)
    ax.set_title('Train data is more than Test data,column:Max_Train')
    plt.savefig('plot.png',dpi=300, bbox_inches='tight')
```

### Visualizing the original data


```python
dataset = rio.open(raster_filename)
# No. of Bands of orthophoto image 
print("No. of Bands of orthophoto image :",dataset.count)
print("Orthophoto Image resolution:","Height:",dataset.height," , width:",dataset.width)
print("Coordinate Reference System :",dataset.crs)
```


```python
dsm = rio.open(DSM)
print("No. of Bands of DSM image :",dsm.count)
print("DSM Image resolution:","Height:",dsm.height," , width:",dsm.width)
print("Coordinate Reference System :",dsm.crs)
```


```python
gli=rio.open(GLI)
print("No. of Bands of Green leaf index(GLI):",gli.count)
print("GLI Image resolution:","Height:",gli.height," , width:",gli.width)
print("Coordinate Reference System :",gli.crs)
```


```python
vari=rio.open(VARI)
print("No. of Bands of Visible Atmospherically Resistant index(VARI):",vari.count)
print("VARI Image resolution:","Height:",vari.height," , width:",vari.width)
print("Coordinate Reference System :",vari.crs)
```

<i>Land cover classes shapefile visualization</i>


```python
shp_file=gpd.read_file(shp_filename)
shp_file_viz = shp_file.loc[:,['Class', 'geometry']]
```


```python
fig, ax = plt.subplots(figsize = (10,10))

shp_file_viz.plot(column='Class',
                              categorical=True,
                              legend=True,  
                              cmap='Set2', ax=ax)
# add a title to the plot

ax.set_title('Land cover classes\nMaima Area, Estonia',
            fontsize=16)

plt.show()

```

<blockquote>Steps for extracting values form RGB orthophoto, DSM and two vegetation indices GLI and VARI respectively</blockquote>



```python
rowcol_tuple = dataset.index(shp_file['geometry'].x, shp_file['geometry'].y)
rowcol = np.asarray(rowcol_tuple).T
```


```python
for i in dataset.indexes:
    band = dataset.read(i)
    # create empty list to save values
    extracted_values = []
    # loop over the "coordinates" in rowcol and extract corresponding value from raster
    if use_25pixels == True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-2:coord[0]+3,coord[1]-2:coord[1]+3])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    elif use_121pixels ==True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-5:coord[0]+6,coord[1]-5:coord[1]+6])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    else:
        for coord in rowcol:
            value = band[coord[0],coord[1]]
            extracted_values.append(value)    
    # add list with extracted values per band into dataframe column
        shp_file['band_' + str(i)] = extracted_values
```


```python
shp_file = shp_file.drop("band_4", axis=1)
shp_file.rename(columns={'band_1': 'red','band_2': 'blue','band_3': 'green'}, inplace=True)
shp_file
```

<i>Extracting DSM values</i> 


```python
# extract data from DSM
rowcol_tuple = dsm.index(shp_file['geometry'].x, shp_file['geometry'].y)
rowcol = np.asarray(rowcol_tuple).T
```


```python
for i in dsm.indexes:
    band = dsm.read(i)
    
    # create empty list to save values
    extracted_values = []
    # loop over the "coordinates" in rowcol and extract corresponding value from raster
    if use_25pixels == True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-2:coord[0]+3,coord[1]-2:coord[1]+3])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    elif use_121pixels ==True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-5:coord[0]+6,coord[1]-5:coord[1]+6])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    else:
        for coord in rowcol:
            value = band[coord[0],coord[1]]
            extracted_values.append(value)    
    # add list with extracted values per band into dataframe column
        shp_file['band_' + str(i)] = extracted_values
    
  
```


```python
shp_file.rename(columns={'band_1': 'DSM'}, inplace=True)
shp_file.head()
```

<i>Green Leaf Index (GLI)</i>


```python
rowcol_tuple = gli.index(shp_file['geometry'].x, shp_file['geometry'].y)
rowcol = np.asarray(rowcol_tuple).T
```


```python
for i in gli.indexes:
    band = gli.read(i)
    
    # create empty list to save values
    extracted_values = []
    # loop over the "coordinates" in rowcol and extract corresponding value from raster
    if use_25pixels == True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-2:coord[0]+3,coord[1]-2:coord[1]+3])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    elif use_121pixels ==True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-5:coord[0]+6,coord[1]-5:coord[1]+6])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    else:
        for coord in rowcol:
            value = band[coord[0],coord[1]]
            extracted_values.append(value)    
    # add list with extracted values per band into dataframe column
        shp_file['band_' + str(i)] = extracted_values
```


```python
shp_file.rename(columns={'band_1': 'GLI'}, inplace=True)
shp_file.head()
```

<i>Visible Atmospherically      Resistant Index(VARI)</i>


```python
rowcol_tuple = vari.index(shp_file['geometry'].x, shp_file['geometry'].y)
rowcol = np.asarray(rowcol_tuple).T
```


```python
for i in vari.indexes:
    band = vari.read(i)
    # create empty list to save values
    extracted_values = []
    # loop over the "coordinates" in rowcol and extract corresponding value from raster
    if use_25pixels == True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-2:coord[0]+3,coord[1]-2:coord[1]+3])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    elif use_121pixels ==True :
        for coord in rowcol:
            value = np.mean(band[coord[0]-5:coord[0]+6,coord[1]-5:coord[1]+6])
            extracted_values.append(value)
        shp_file['band_' + str(i)] = extracted_values
    else:
        for coord in rowcol:
            value = band[coord[0],coord[1]]
            extracted_values.append(value)    
    # add list with extracted values per band into dataframe column
        shp_file['band_' + str(i)] = extracted_values
```


```python
shp_file.rename(columns={'band_1': 'VARI'}, inplace=True)
shp_file.head()
```


```python
dataset121 = os.path.join(path,'dataset121.csv')
shp_file.to_csv(dataset121, index=False)
```


```python
#if use_25pixels or use_121pixels == True:
   # shp_file['VARI']=(shp_file['green']-shp_file['red'])/shp_file['green']+shp_file['red']-shp_file['blue']
    #shp_file['GLI']=((2*shp_file['green'])-shp_file['red']-shp_file['blue'])/((2*shp_file['green'])+shp_file['red']+shp_file['blue'])
   # shp_file.head()
```


```python
 shp_file.head()
```


```python
sns.set(style="ticks", color_codes=True);
# Create a custom color palete
palette = sns.xkcd_palette(['dark blue', 'green', 'purple', 'orange','dark green','gold','blue','red'])
# Make the pair plot with a some aesthetic changes
if use_different_split==True:
    shp_file_new=shp_file[['Class','geometry','red','blue','green','DSM','GLI','VARI']]
    fig=sns.pairplot(shp_file_new, hue = 'Class', diag_kind = 'kde', palette= palette, plot_kws=dict(alpha = 0.7), diag_kws=dict(shade=True))
    fig.savefig("pairplot.png")
else:
    fig=sns.pairplot(shp_file, hue = 'Class', diag_kind = 'kde', palette= palette, plot_kws=dict(alpha = 0.7), diag_kws=dict(shade=True))
    fig.savefig("pairplot.png")
    
   
```

## Counting the distribution of labels in dataframe 


```python
count_class = pd.DataFrame(shp_file.groupby('Class').size())
ax = shp_file.Class.value_counts().plot(kind='bar', title="Distribution of labels in Data")
ax.set_xlabel("Class type")
ax.set_ylabel("Count")
ax.legend(title='Landcover classes')
```

## Extracting train and test values

<blockquote>In order to have train and test data separated for further classification process the test and train data has been splitted by assigining 30% to Test data and 70% to Train data.Data has been stratified by class label which here is "label"</blockquote>



```python
if use_different_split==True:
    shp_file['Random'].dtypes
    shp_file['Random'] = shp_file['Random'].astype(int)
    X=shp_file.loc[shp_file['Random'] == 1]# Extracting Train data
    y=shp_file.loc[shp_file['Random'] == 0]# Extracting Test data
    # Finding data with null values
    print(X.isnull().sum())
    
else:
    # Train and Test split
    label=shp_file.iloc[:,0]
    Train, Test = train_test_split(shp_file,test_size=0.3,stratify=label,random_state=42)
    # Finding data with null values
    print(Train.isnull().sum())
```

<blockquote> By selecting the definite columns in our data base we made a dataframe with same structure for every data input with different spatial distribution or with normal distribution of train and test. The finalized dataframe will be converted to csv file as a input data for the rest of the analyzation</blockquote>

##### In case of selecting the different spatial distribution 



```python
if use_different_split==True:
    X=X[['Class','geometry','red','blue','green','DSM','GLI','VARI']]
    y=y[['Class','geometry','red','blue','green','DSM','GLI','VARI']]
```


```python
# Visualizing the dataset after selection
X.head()
```


```python
y.head()
```

## Visualizing Train and Test data


```python
if use_different_split==True: 
    count_class = pd.DataFrame(X.groupby('Class').size())
    ax = X.Class.value_counts().plot(kind='bar', title="Distribution of labels of train data with differnt spatial distribution")
    ax.set_xlabel("Class type")
    ax.set_ylabel("Count")
    ax.legend(title='Landcover classes')
else:
    
    count_class = pd.DataFrame(Train.groupby('Class').size())
    ax = Train.Class.value_counts().plot(kind='bar', title="Distribution of labels in train data")
    ax.set_xlabel("Class type")
    ax.set_ylabel("Count")
    ax.legend(title='Landcover classes')
    
```


```python
if use_different_split==True: 
    count_class = pd.DataFrame(y.groupby('Class').size())
    ax = y.Class.value_counts().plot(kind='bar', title="Distribution of labels of test data with differnt spatial distribution")
    ax.set_xlabel("Class type")
    ax.set_ylabel("Count")
    ax.legend(title='Landcover classes')
else:
    
    count_class = pd.DataFrame(Test.groupby('Class').size())
    ax = Test.Class.value_counts().plot(kind='bar', title="Distribution of labels in test data")
    ax.set_xlabel("Class type")
    ax.set_ylabel("Count")
    ax.legend(title='Landcover classes')
```

## Extracting values of train and test data to csv


```python
#Each time creating the csv files based on different spatial sampling the name of the csv file should change.
```


```python
if use_different_split==True:
    train_Spatial_d3 = os.path.join(path,'train_Spatial_Random_p121.csv')
    X.to_csv(train_Spatial_d3, index=False)
    test_Spatial_d3 = os.path.join(path,'test_Spatial_Random_p121.csv')
    y.to_csv(test_Spatial_d3, index=False)
else:
    Train_split_70 = os.path.join(path,'Train_split_70_121p.csv')
    Train.to_csv(Train_split_70, index=False)
    Test_split_30 = os.path.join(path,'Test_split_30_121p.csv')
    Test.to_csv(Test_split_30, index=False)
   
```

#### Different spatial distribution (Spatial sampling)  and Distance plots 


```python
dfrp13 = pd.read_csv('halfp11_distance.csv',sep=';')
dfrp13.rename(columns={'rvalue_1': 'Distance'}, inplace=True)
```


```python
df_L3 = np.where(dfrp13['Predicted'] == dfrp13['Class'], True, False)
df_L3=pd.DataFrame(data=df_L3)
df_R3= pd.concat([df_L3, dfrp13],axis = 1)
df_R3.rename(columns={0: 'labels'}, inplace=True)
df_R3 = df_R3.drop("Class", axis=1)
df_R3 = df_R3.drop("Predicted", axis=1)
```


```python
X=df_R3.loc[df_R3['labels'] == True]
y=df_R3.loc[df_R3['labels'] == False]
```


```python
X.to_csv('TrueLablem.csv')
y.to_csv('FalseLablem.csv')
```


```python
plt.hist(y['Distance'],edgecolor='black',linewidth=1,bins=30, rwidth=3.12,range=[0,105],alpha=0.7,label="False")
plt.hist(X['Distance'],edgecolor='black',linewidth=1,bins=30, rwidth=3.12,range=[0,105],label="True",alpha=0.7)

plt.title('Distance pattern (Train and Test)', fontsize=14)
plt.xlabel('Distance(Random distribution)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
labels= ["Fasle"]
plt.legend(loc='upper right')
plt.savefig('random_histogram.png')
```


```python
plt.hist(y['Distance'],edgecolor='black',linewidth=1,bins=33, rwidth=7.35,range=[0,255],alpha=0.7,label="False")
plt.hist(X['Distance'],edgecolor='black',linewidth=1,bins=33, rwidth=7.35,range=[0,255],label="True",alpha=0.7)
plt.title('Distance pattern (Train and Test)', fontsize=14)
plt.xlabel('Distance(Equal distribution)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
labels= ["Fasle"]
plt.legend(loc='upper right')
plt.savefig('equal_histogram.png')
```


```python
plt.hist(y['Distance'],edgecolor='black',linewidth=1,bins=23, rwidth=13.03,range=[0,350],alpha=0.7,label="False")
plt.hist(X['Distance'],edgecolor='black',linewidth=1,bins=23, rwidth=13.03,range=[0,350],alpha=0.7,label="True")
plt.title('Distance pattern (Train and Test)', fontsize=14)
plt.xlabel('Distance(Biased distribution)', fontsize=12)
plt.ylabel('Counts', fontsize=12)
labels= ["Fasle"]
plt.legend(loc='upper right')
plt.savefig('biased_histogram.png')
```


```python

```
