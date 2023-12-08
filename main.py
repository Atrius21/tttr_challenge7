# Importing the required libraries
import cartopy.crs as ccrs
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import glob
import rasterio
import re
from datetime import datetime
import numpy as np
import geopandas as gpd
from shapely import wkt
from sklearn.neighbors import BallTree
import geog
import shapely.geometry as geometry

import folium as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_selector as selector

from sklearn.linear_model import LogisticRegression, LinearRegression, Perceptron
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error




# mapping the dictionary column size to reduce the dataframe size
col_type = {
    'band': 'int16',
    'y':'float16',
    'x': 'float16',
    'spatial_ref': 'int16',
    'band_data': 'float32'
}



"""
Data Cleaning 
- converting img into tabular data
- reduce memory usage changing column types

"""
df_cloud_fraction = pd.DataFrame()
df_no2_flux  = pd.DataFrame()
df_no2 = pd.DataFrame()
df_boundary_layer_height = pd.DataFrame()
df_relative_humidity = pd.DataFrame()
df_solar_radiation = pd.DataFrame()
df_temperature = pd.DataFrame()
df_uwind = pd.DataFrame()
df_vwind = pd.DataFrame()

def convert_to_dataframe(filename):
    dt_ = re.findall(r'_(\d{1,8})', filename)[0]
    temp = xr.open_dataarray(item).to_dataframe().reset_index()
    temp['partition_date'] = dt_
    temp = temp.astype(col_type)
    return temp

for item in glob.glob('./data/NO2flux/*'):
    temp_df = convert_to_dataframe(item)
    df_no2_flux = pd.concat([temp_df, df_no2_flux], ignore_index = True)


for item in glob.glob('./data/satellite/NO2/*'):
    temp_df = convert_to_dataframe(item)
    df_no2 = pd.concat([temp_df, df_no2],ignore_index = True)

for item in glob.glob('./data/satellite/cloud_fraction/*'):
    temp_df = convert_to_dataframe(item)
    df_cloud_fraction = pd.concat([temp_df, df_cloud_fraction],ignore_index = True)

for item in glob.glob('./data/weather/boundary_layer_height/*'):
    temp_df = convert_to_dataframe(item)
    df_boundary_layer_height = pd.concat([temp_df, df_boundary_layer_height],ignore_index = True)


for item in glob.glob('./data/weather/relative_humidity/*'):
    temp_df = convert_to_dataframe(item)
    df_relative_humidity = pd.concat([temp_df, df_relative_humidity],ignore_index = True)


for item in glob.glob('./data/weather/solar_radiation/*'):
    temp_df = convert_to_dataframe(item)
    df_solar_radiation = pd.concat([temp_df, df_solar_radiation],ignore_index = True)


for item in glob.glob('./data/weather/temperature/*'):
    temp_df = convert_to_dataframe(item)
    df_temperature = pd.concat([temp_df, df_temperature],ignore_index = True)


for item in glob.glob('./data/weather/wind/uwind_*'):
    temp_df = convert_to_dataframe(item)
    df_uwind = pd.concat([temp_df, df_uwind],ignore_index = True)

for item in glob.glob('./data/weather/wind/vwind_*'):
    temp_df = convert_to_dataframe(item)
    df_vwind = pd.concat([temp_df, df_vwind],ignore_index = True)


"""
Converts the partition date into datetime so we can select based on the dates
"""

df_cloud_fraction['partition_date'] = df_cloud_fraction['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_no2_flux['partition_date'] = df_no2_flux['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_no2['partition_date'] = df_no2['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)


df_boundary_layer_height['partition_date'] = df_boundary_layer_height['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_relative_humidity['partition_date'] = df_relative_humidity['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_solar_radiation['partition_date'] = df_solar_radiation['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_temperature['partition_date'] = df_temperature['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_uwind['partition_date'] = df_uwind['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)

df_vwind['partition_date'] = df_vwind['partition_date'].apply(
    lambda g: datetime.strptime(g, '%Y%m%d')
)


"""
Selecting a certain period of time for the training data

"""
strt_dt = datetime.strptime("2019-03-01", '%Y-%m-%d')
end_dt = datetime.strptime("2020-03-01", '%Y-%m-%d')

df = {}
for item in glob.glob('./*_dt.pkl'):
    str_ = item.split('.')[1].replace('/', '')
    temp = pd.read_pickle(item)
    temp = temp.loc[
        temp['partition_date'].between(strt_dt, end_dt)
    ]
    # choosing between dates
    df[str_] = temp





"""
Reading the Taiwan Powerplant data and emissions
"""

df_tw_power = pd.read_csv('./Taiwan_powerplants.csv')
df_tw_emissions = pd.read_csv('./Taiwan_nox_emissions.csv')

# converts the latlng values to geopandas readable format
df_tw_power['geom'] = df_tw_power.geom.apply(wkt.loads)
df_tw_power['lat'] = df_tw_power.geom.apply(lambda g: g.y)
df_tw_power['lng'] = df_tw_power.geom.apply(lambda g: g.x)
df_tw_power['latlng'] = list(zip(df_tw_power['lat'], df_tw_power['lng']))




# Using balltree index to find the nearest power plants and it's distance
 # Ball Tree Radius query
RADIANT_TO_KM_CONSTANT = 6367
query_list = dict(zip(df_tw_power['facility_id'], df_tw_power['latlng']))

class BallTreeIndex:
    def __init__(self,lat_longs):
        self.lat_longs = np.radians(lat_longs)
        self.ball_tree_index =BallTree(self.lat_longs, metric='haversine')

    def query_radius(self,query,radius):
        radius_km = radius/1e3
        radius_radian = radius_km / RADIANT_TO_KM_CONSTANT 
        query = np.radians(np.array([query]))
        indices = self.ball_tree_index.query_radius(query,r=radius_radian)     
        return indices[0]
    
    def query(self, query):
        query = np.radians(np.array([query]))
        dist, ind = self.ball_tree_index.query(query, k=2) 
        return ind[0][1], dist[0][1]

latlons = [v for k,v in query_list.items()]
    
p1 = BallTreeIndex(latlons)


# Functions to retrieve the nearest stations and distance
def return_sites(lat_longs, radius):
    """
    Returns the nearest latlongs based on 
    input radius
    """
    input_latlong=lat_longs # input your latlong e.g lat,long
    radius = radius # radius in metres
    id_site = [list(query_list)[x] for x in list(p1.query_radius(input_latlong, radius))]
    
    #returns the site_ids within the specify radius of your latlong
    return id_site


def nearest_site(lat_longs):
##========================================##
    
    input_latlong=lat_longs # input your latlong e.g lat,long
    # id_site = [list(query_list)[x] for x in list(p1.query(input_latlong))]
    idx = p1.query(input_latlong)[0]
    id_site = list(query_list)[idx]

    # radius
    dist_rad = p1.query(input_latlong)[1]
    dist_km = dist_rad * RADIANT_TO_KM_CONSTANT

    
    
    return id_site, dist_km

def return_latlon(lat_longs, radius):
##========================================##

    # getting the queried site_ids
    input_latlong=lat_longs # input your latlong e.g lat,long
    radius = radius # radius in metres
    id_tag = [list(query_list.values())[x] for x in list(p1.query(input_latlong))]
    
    #returns the site_ids within the specify radius of your latlong
    return id_tag



df_tw_power['nearest_station'] = df_tw_power.apply(
    lambda g: nearest_site(g.latlng)[0], axis = 1
)

df_tw_power['nearest_station_km'] = df_tw_power.apply(
    lambda g: nearest_site(g.latlng)[1], axis = 1
)


"""
Merging the data into one singular table, each represented by the pixel locations
"""

map_dict = {
    'df_uwind_dt': 'uwind'
    , 'df_cloud_fraction_dt': 'cloud_fraction'
    , 'df_no2_dt': 'no2'
    , 'df_boundary_layer_height_dt': 'layer_height'
    , 'df_relative_humidity_dt': 'relative_humidity'
    , 'df_solar_radiation_dt': 'solar_radiation'
    , 'df_temperature_dt': 'temperature'
    , 'df_no2_flux_dt':'no2_flux'
    , 'df_vwind_dt': 'vwind'
}

join_key = [
    'y','x','partition_date'
]

for k,v in df.items():
    df[k] = df[k].groupby(join_key).agg({
    'band_data'  : 'mean'
    }).reset_index()



for k,v in df.items():
    # df[k] = v.drop(columns = ['spatial_ref', 'band'])
    df[k] = df[k].rename(columns = {
        'band_data': map_dict[k]
    })




# setting main_df as the main dataframe that contains all the climate data represented by the pixe locations across the days
main_df = df['df_uwind_dt'].merge(
    df['df_vwind_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_cloud_fraction_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_no2_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_boundary_layer_height_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_relative_humidity_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_solar_radiation_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_temperature_dt']
    , on = join_key
    , how = 'outer'
).merge(
    df['df_no2_flux_dt']
    , on = join_key
    , how = 'outer'
)

"""
Creating geopandas readable geometry. This is where the clipping happens. 
We will only look at data within 10KM of each power plants which has no other plants nearby
"""

main_df['geometry'] = gpd.points_from_xy(main_df['x'], main_df['y'])
main_df_gdf = gpd.GeoDataFrame(main_df, geometry='geometry')
main_df_gdf.crs = "EPSG:4326"


radius_list = [
    'circle_10km', 'tier_20km',
       'tier_30km', 'tier_40km', 'tier_50km'
]

# This 
clipped_dict = {}
for item in radius_list:
        
    temp = gpd.GeoDataFrame(df_tw_power, geometry = item)
    temp.crs = "EPSG:4326"

    clipped_dict[item] = temp.sjoin(
        main_df_gdf
        , predicate = 'intersects'
    ).reset_index(drop = True)



join_key = [
    'facility_id', 'partition_date','x','y'
]

metric_key = [
    'uwind', 'vwind', 'cloud_fraction', 'no2',
       'layer_height', 'relative_humidity', 'solar_radiation', 'temperature',
       'no2_flux'
]

# Essentially Clipping the data for every feature, 
for k,v in clipped_dict.items():    
    clipped_dict[k].columns = [x + '_{}'.format(k) if x in metric_key else x for x in clipped_dict[k] ]
    clipped_dict[k] = clipped_dict[k].drop(columns = [
        'data_source', 'name', 'iso2', 'geom', 'lat', 'lng',
       'latlng', 'nearest_station', 'nearest_station_km', 'circle_10km', 'circle_20km', 'circle_30km',
       'circle_40km', 'circle_50km', 'tier_20km', 'tier_30km', 'tier_40km',
       'tier_50km', 'index_right',
        # 'y','x'
    ])

    
# inner joining isolated power plants
facility_split = pd.DataFrame()
new_cols = [
    'facility_id', 'nearest_station', 'nearest_station_km'
]

temp = df_tw_power.loc[
    (df_tw_power['nearest_station_km']<=20)
, new_cols].copy()
temp['near'] = 'yes'
facility_split = pd.concat([facility_split, temp], ignore_index=True)

temp = df_tw_power.loc[
    (df_tw_power['nearest_station_km']>20)
, new_cols].copy()
temp['near'] = 'no'
facility_split = pd.concat([facility_split, temp], ignore_index=True)

df_truth_merged = df_truth.merge(
    facility_split
    , on = ['facility_id']
    , how = 'left'
)

df_truth_merged = df_truth_merged.loc[
    df_truth_merged['near']=='no'
]

df_truth_merged = df_truth_merged.rename(columns = {'datetime': 'partition_date'})





"""
Prepping the data for modelling and training
"""
X_train = clipped_dict['circle_10km'].copy()
no2_mean = X_train['no2_circle_10km'].mean()
X_train['no2_circle_10km'] = X_train['no2_circle_10km'].fillna(no2_mean)

cloud_mean = X_train['cloud_fraction_circle_10km'].mean()
X_train['cloud_fraction_circle_10km'] = X_train['cloud_fraction_circle_10km'].fillna(cloud_mean)

X_train = X_train.drop(columns = ['no2_flux_circle_10km'])

X_train['partition_date'] = X_train['partition_date'].apply(lambda g: g.strftime("%Y-%m-%d"))

X_train = X_train.merge(
    df_truth_merged
    , on = ['facility_id', 'partition_date']
    , how = 'left'
)


cols_drop = [
    'facility_id'
    #  , 'y', 'x'
    ,'partition_date','data_source','poll','unit', 'near', 'nearest_station', 'nearest_station_km', 'value'
    
]

X_train = X_train.dropna(subset = ['value', 'no2_circle_10km'])

y_train = X_train['value'].copy()
X_train = X_train.drop(columns = cols_drop)


# test set data
df2 = pp.preprocess("2020-03-02", '2020-12-31')

X_test = df2['circle_10km'].copy()
no2_mean = X_test['no2_circle_10km'].mean()
X_test['no2_circle_10km'] = X_test['no2_circle_10km'].fillna(no2_mean)

cloud_mean = X_test['cloud_fraction_circle_10km'].mean()
X_test['cloud_fraction_circle_10km'] = X_test['cloud_fraction_circle_10km'].fillna(cloud_mean)

X_test = X_test.drop(columns = ['no2_flux_circle_10km'])

X_test['partition_date'] = X_test['partition_date'].apply(lambda g: g.strftime("%Y-%m-%d"))

X_test = X_test.merge(
    df_truth_merged
    , on = ['facility_id', 'partition_date']
    , how = 'left'
)

cols_drop = [
    'facility_id'
    # , 'y', 'x'
    ,'partition_date','data_source','poll','unit', 'near', 'nearest_station', 'nearest_station_km', 'value'
    
]

X_test = X_test.dropna(subset = ['value', 'no2_circle_10km'])
y_test = X_test['value'].copy()
X_test = X_test.drop(columns = cols_drop)



# shap analysis
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

explainer = shap.TreeExplainer(gbr)


#apply the preprocessing to x_test
shap_values = explainer.shap_values(X_test)

#plot the feature importance
shap.summary_plot(shap_values, X_test, plot_type='dot', show = False
                 , max_display= 30)

# this needs to be here because shap isn't optimized for newer versions of matplotlib yet
plt.gcf().axes[-1].set_aspect(100)
plt.gcf().axes[-1].set_box_aspect(100)



# modelling 
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
rfr = RandomForestRegressor(n_estimators=1000, max_depth=4)
gbr =GradientBoostingRegressor(n_estimators=1000, learning_rate=0.05)

models = [
    xgb, rfr, gbr
]
for model in models:
    model.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_pred_rfr = rfr.predict(X_test)
y_pred_gbr = gbr.predict(X_test)


pred_dict = {
    'Xgb': y_pred_xgb
    , 'RFR': y_pred_rfr
    , 'GBR': y_pred_gbr
}

for k,v in pred_dict.items():
    print('{}: {}'.format(k, mean_squared_error(y_test, v, squared= False)))