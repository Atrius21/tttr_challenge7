import pandas as pd
from datetime import datetime
import geopandas as gpd
import glob

def preprocess(strt_dt, end_dt):
    strt_dt = datetime.strptime(strt_dt, '%Y-%m-%d')
    end_dt = datetime.strptime(end_dt, '%Y-%m-%d')

    df = {}
    for item in glob.glob('./*_dt.pkl'):
        str_ = item.split('.')[1].replace('/', '')
        temp = pd.read_pickle(item)
        temp = temp.loc[
            temp['partition_date'].between(strt_dt, end_dt)
        ]
        # choosing between dates
        df[str_] = temp






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





    # setting df_cloud_fraction_dt as the main df for merging
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



    main_df['geometry'] = gpd.points_from_xy(main_df['x'], main_df['y'])
    main_df_gdf = gpd.GeoDataFrame(main_df, geometry='geometry')
    main_df_gdf.crs = "EPSG:4326"


    radius_list = [
        'circle_10km', 'tier_20km',
        'tier_30km', 'tier_40km', 'tier_50km'
    ]




# df_tw_power.to_pickle('df_tw_power.pkl')
    df_tw_power = pd.read_pickle('df_tw_power.pkl')

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

    for k,v in clipped_dict.items():
        
        clipped_dict[k].columns = [x + '_{}'.format(k) if x in metric_key else x for x in clipped_dict[k] ]


    for k,v in clipped_dict.items():
        clipped_dict[k] = clipped_dict[k].drop(columns = [
            'data_source', 'name', 'iso2', 'geom', 'lat', 'lng',
        'latlng', 'nearest_station', 'nearest_station_km', 'circle_10km', 'circle_20km', 'circle_30km',
        'circle_40km', 'circle_50km', 'tier_20km', 'tier_30km', 'tier_40km',
        'tier_50km', 'index_right',          
        ])

    return clipped_dict