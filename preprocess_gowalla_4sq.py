from tqdm import tqdm
from setting import SetParameter
import os.path as osp
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.neighbors import NearestNeighbors

args = SetParameter()


def encode_cat_id(df, column):
    label_encoder = LabelEncoder()
    df['CatId'] = label_encoder.fit_transform(df[column].values.tolist())


def encode_user_id(df, column):
    label_encoder = LabelEncoder()
    df['UserId'] = label_encoder.fit_transform(df[column].values.tolist())


def preprocess_dataset():
    data_path = osp.join('data', args.dataset_name)
    sample_file = osp.join(data_path, 'sample.csv')
    raw_poi_file = osp.join(data_path, 'raw_poi.csv')
    poi_file = osp.join(data_path, 'poi_sample.csv')
    nearest_poi_idx_file = osp.join(data_path, 'nearest_poi_idx.csv')
    nearest_poi_dis_file = osp.join(data_path, 'nearest_poi_dis.csv')

    print(f'preprocess:{args.dataset_name}')
    if args.dataset_name == 'gowalla':
        raw_file = 'checkins-gowalla.txt'
        raw_pois = pd.read_csv(osp.join(data_path, 'gowalla_spots_subset1.csv'))[
            ['lng', 'lat', 'spot_categories']]
        raw_pois.columns = ['Longitude', 'Latitude', 'RawCatId']  # 注意经纬度顺序
        raw_pois['Latitude'] = raw_pois['Latitude'].round(8)
        raw_pois['Longitude'] = raw_pois['Longitude'].round(8)
    else:
        raw_file = 'checkins-4sq.txt'
        raw_pois = pd.read_csv(osp.join(data_path, 'raw_POIs.txt'), sep='\t', encoding='latin-1', header=None,
                               usecols=[1, 2, 3])
        raw_pois.columns = ['Latitude', 'Longitude', 'RawCatId']
        raw_pois['Latitude'] = raw_pois['Latitude'].round(8)
        raw_pois['Longitude'] = raw_pois['Longitude'].round(8)
    # 去除原始pois经纬度一致但是PoiId不一致问题
    raw_pois.drop_duplicates(subset=['Latitude', 'Longitude'], keep='first', ignore_index=True, inplace=True)
    # %% ===============================
    # 处理轨迹，去除小于100的用户
    df = pd.read_csv(osp.join(data_path, raw_file), sep='\t', encoding='latin-1', header=None)
    df.drop_duplicates(keep='first', inplace=True, ignore_index=True)  # 删除重复数据
    df.columns = ['UserId', 'UTCTime', 'Latitude', 'Longitude', 'RawPoiId']
    user_count = df.groupby('UserId')['RawPoiId'].count().reset_index()
    df = df[df['UserId'].isin(user_count[user_count['RawPoiId'] >= args.min_user_freq]['UserId'])]
    df.sort_values(by=['UserId', 'UTCTime'], ascending=True, inplace=True, ignore_index=True)
    # %% ===============================
    # 创建NearestNeighbors模型   # 给轨迹中需要的poi匹配cat
    df_pois = df[['Latitude', 'Longitude']].drop_duplicates(ignore_index=True)
    df_pois['Latitude'] = df_pois['Latitude'].round(8)
    df_pois['Longitude'] = df_pois['Longitude'].round(8)
    df_pois = pd.merge(df_pois, raw_pois, on=['Latitude', 'Longitude'], how='left')
    # 处理gowalla中某些poi没有对应的cat
    if args.dataset_name == 'gowalla':
        raw_pois_neighbors = NearestNeighbors(n_neighbors=2, metric='haversine')
        raw_pois_neighbors.fit(np.radians(raw_pois[['Latitude', 'Longitude']].values))
        no_catid_pois = df_pois[df_pois['RawCatId'].isnull()]
        nearest_cat_poi_dis, nearest_cat_poi_idx = raw_pois_neighbors.kneighbors(
            np.radians(no_catid_pois[['Latitude', 'Longitude']].values))
        unmatched_indices = no_catid_pois.index
        for i in tqdm(range(len(unmatched_indices))):
            nearest_index = nearest_cat_poi_idx[i, 0]
            index = unmatched_indices[i]
            df_pois.loc[index, 'RawCatId'] = raw_pois.loc[nearest_index, 'RawCatId']
    # %% ===============================
    # 在df_pois中查找最近的200个poi
    df_pois_neighbors = NearestNeighbors(n_neighbors=200, metric='haversine')
    df_pois_neighbors.fit(np.radians(df_pois[['Latitude', 'Longitude']].values))
    nearest_poi_dis, nearest_poi_idx = df_pois_neighbors.kneighbors(
        np.radians(df_pois[['Latitude', 'Longitude']].values))  # 角度值转换成弧度制
    nearest_poi_dis = nearest_poi_dis * 6371  # 地球半径
    # 编码,合并
    print('encode id')
    encode_cat_id(df_pois, 'RawCatId')
    encode_user_id(df, 'UserId')
    df_pois['PoiId'] = range(len(df_pois))

    df = pd.merge(df, df_pois, on=['Latitude', 'Longitude'], how='left')
    df.sort_values(by=['UserId', 'UTCTime'], ascending=True, inplace=True, ignore_index=True)

    print('save')
    raw_pois.to_csv(raw_poi_file, index=False)
    df_pois.to_csv(poi_file, index=False)
    df.to_csv(sample_file, index=False)
    pd.DataFrame(nearest_poi_dis).to_csv(nearest_poi_dis_file, index=False)
    pd.DataFrame(nearest_poi_idx).to_csv(nearest_poi_idx_file, index=False)
    print(len(df))
    # %% ===============================================================================================================


if __name__ == '__main__':
    preprocess_dataset()
