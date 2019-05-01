###
# Chapter 2
###

#%% define housing path
import os
HOUSING_PATH = os.path.join("datasets", "housing")

#%% function for loading data - ONLY RUN ONCE

import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

#%% load data - ONLY RUN ONCE
fetch_housing_data()

#%% load data into Pandas
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

#%% Look at the data structure
housing = load_housing_data()
housing.head()


#%%
housing.info()

#%%
housing['ocean_proximity'].value_counts()

#%%
housing.describe()

#%%
import matplotlib.pyplot as plt
import pandas as pd
#%%
housing.hist(bins=50, figsize=(20,15))
plt.show()

#%%
import numpy as np
housing['income_cat'] = np.ceil(housing['median_income'] /1.5)
housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)

#%%
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#%%
housing['income_cat'].value_counts() / len(housing)

#%%
for set in (strat_train_set, strat_test_set):
    set.drop(['income_cat'], axis=1, inplace=True)

#%%
housing = strat_train_set.copy()

#%%
housing.plot(kind='scatter', x='longitude', y='latitude')

#%%
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

#%%
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population',
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()

#%%
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
import pandas as pd
from pandas.tools.plotting import scatter_matrix

attributes =['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12, 8))

#%%
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']
housing['population_per_household'] = housing['population'] / housing['households']


#%%
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)

#%%
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

#%%
