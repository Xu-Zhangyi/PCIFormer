from setting import SetParameter
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from utils import deduplicate_matrix, sparse_matrix_to_tensor
import os.path as osp
import os
import time

args = SetParameter()


def build_user_data():
    user_poi_pref = np.zeros([user_count, poi_count])  # (U,P)
    for i in tqdm(range(len(tra_user_id)), desc='build user2poi graph'):
        train_idx = int(len(tra_user_id[i]) * 0.8)
        np.add.at(user_poi_pref, (tra_user_id[i][:train_idx], tra_poi_id[i][:train_idx]), 1)

    print('normalized user poi pref')
    user_poi_pref = torch.tensor(user_poi_pref, dtype=torch.float32)
    num_user_poi_pref = torch.nonzero(user_poi_pref).size(0)  # gowalla poi 595089,user 7768 ;4sq poi 1672225,user 45331
    user_poi_pref_val, user_poi_pref_idx = user_poi_pref.topk(args.topk_user_poi_pref, largest=True)
    num_user_poi_pref_val = torch.nonzero(user_poi_pref_val).size(0)
    #gowalla:(40)275448,(60)364727,(80)423317,(100)462145,(120)489009
    #4sq:(20)759080,(30)1038600,(40)1253796,(50)1404491,(60)1501253
    row_sums = torch.sum(user_poi_pref_val, dim=-1, keepdim=True)
    user_poi_pref_val = user_poi_pref_val / row_sums
    user_poi_pref_mask = torch.where(user_poi_pref_val != 0, 0, float('-inf'))

    user_data = {
        'user_poi_pref_idx': user_poi_pref_idx,
        'user_poi_pref_val': user_poi_pref_val,
        'user_poi_pref_mask': user_poi_pref_mask}

    torch.save(user_data, osp.join(data_path, 'user', 'topk_user_poi_pref_' + str(args.topk_user_poi_pref) + '.pt'))


def build_cat_data():
    cat_transition = np.zeros([cat_count, cat_count])  # (C,C)
    cat_poi_mask = torch.full([cat_count, poi_count], float('-inf'))  # (C,P)

    for i in tqdm(range(len(tra_user_id)), desc='build cat transition graph'):
        train_idx = int(len(tra_poi_id[i]) * 0.8)
        np.add.at(cat_transition, (tra_cat_id[i][:train_idx - 1], tra_cat_id[i][1:train_idx]), 1)

    row_indices = df_pois['CatId'].to_list()
    col_indices = df_pois['PoiId'].to_list()
    cat_poi_mask[row_indices, col_indices] = 0
    print('normalized cat transition')
    cat_transition = torch.tensor(cat_transition)
    row_sums1 = torch.sum(cat_transition, dim=-1, keepdim=True)
    row_sums1 = torch.where(row_sums1 == 0, 1, row_sums1)
    cat_transition = cat_transition / row_sums1

    cat_poi_mask = coo_matrix(cat_poi_mask)  # (C,P)
    cat_transition = coo_matrix(cat_transition)  # (C,C)
    cat_data = {
        'cat_transition': cat_transition,
        'cat_poi_mask': cat_poi_mask}
    torch.save(cat_data, osp.join(data_path, 'cat', 'cat.pt'))


def build_dis_threshold_neighbor_data():
    poi_distance = torch.zeros([poi_count, poi_count], dtype=torch.float32)
    distance = torch.exp(-dis_poi_dis).float()
    distance[dis_poi_dis > args.dis_threshold] = 0
    poi_distance[torch.arange(poi_count).unsqueeze(-1), dis_poi_idx] = distance
    num_poi_distance = torch.nonzero(poi_distance).size(0)
    # ================================================
    # tran gowalla:941953; 4sq:2309888

    # dis
    # gowalla:(0.2)1000651,(0.5)3096172,(1.0)5546055 (1.5)7136205 (2.5)9373396
    # 4sq:(0.5)737597,(1.0)1569746,(1.5)2421613,(2.0)3197064,(2.5)3881401,(5.0)6207682
    # ================================================
    poi_dis_neighbor = dis_poi_idx[:, :args.topk_poi_distance]
    poi_dis_neighbor_mask = ~(dis_poi_dis[:, :args.topk_poi_distance] > args.dis_threshold)
    num_poi_dis_neighbor_mask = torch.sum(poi_dis_neighbor_mask)
    # 4sq_5:(1.5)268358
    # 4sq_10:(1.5)463645
    # 4sq_15:(1.5)626116
    # 4sq_20:(1.5)767012
    # 4sq_25:(1.5)892396
    # gowalla_5:(0.5)414482
    # gowalla_10:(0.5)694300
    # gowalla_15:(0.5)915859
    # gowalla_20:(0.5)1100933
    # gowalla_25:(0.5)1260011
    poi_dis_neighbor_mask = torch.where(poi_dis_neighbor_mask, 0, float('-inf'))

    poi_distance = coo_matrix(poi_distance)
    poi_dis_neighbor_mask = coo_matrix(poi_dis_neighbor_mask)
    dis_neighbor_data = {
        'poi_distance': poi_distance,
        'poi_dis_neighbor': poi_dis_neighbor,
        'poi_dis_neighbor_mask': poi_dis_neighbor_mask}
    torch.save(dis_neighbor_data, osp.join(data_path, 'dis_neighbor',
                                           'dis_threshold_' + str(args.dis_threshold) +
                                           'topk_poi_distance_' + str(args.topk_poi_distance) + '.pt'))


def build_tran_neighbor_data():
    poi_transition = np.zeros([poi_count, poi_count])  # (P,P)
    cat_transition = np.zeros([cat_count, cat_count])  # (C,C)
    cat_poi_relationship = torch.zeros([cat_count + 1, poi_count])  # (C,C)

    row_indices = df_pois['CatId'].to_list()
    col_indices = df_pois['PoiId'].to_list()
    cat_poi_relationship[row_indices, col_indices] = 1

    for i in tqdm(range(len(tra_user_id)), desc='build transition graph'):
        train_idx = int(len(tra_poi_id[i]) * 0.8)
        np.add.at(poi_transition, (tra_poi_id[i][:train_idx - 1], tra_poi_id[i][1:train_idx]), 1)
        np.add.at(cat_transition, (tra_cat_id[i][:train_idx - 1], tra_cat_id[i][1:train_idx]), 1)
    cat_transition = torch.tensor(cat_transition)
    # %% =============================================================================================================
    print('normalized poi transition')
    poi_transition = torch.tensor(poi_transition, dtype=torch.float32)
    # num_poi_transition = torch.nonzero(poi_transition).size(0)  # gowalla:941953;4sq: 2309888

    row_sums1 = torch.sum(poi_transition, dim=1, keepdim=True)
    row_sums1 = torch.where(row_sums1 == 0, 1, row_sums1)
    poi_transition = poi_transition / row_sums1
    # # %% ===============================
    poi_tran_neighbor_val, poi_tran_neighbor_idx = (poi_transition + torch.eye(poi_count)).topk(
        args.topk_poi_transition, largest=True)
    num_poi_tran_neighbor_val = torch.nonzero(poi_tran_neighbor_val).size(0)
    # gowalla:(10)716379 (20)869914 (30)923516 (40)950620 (50)967471
    # 4sq:(10)561020(20)933657(30)1199801(40)1388277(50)1522035
    # %% =================================
    if args.tran_dis_threshold != 0:
        print('build poi tran neighbor')
        poi_tran_neighbor_idx[poi_tran_neighbor_val == 0] = -1
        cat_tran_neighbor_val, cat_tran_neighbor_idx = cat_transition.topk(100, largest=True)  # (C,10)

        cat_tran_neighbor_idx[cat_tran_neighbor_val < 80] = cat_count
        num_cat_tran_neighbor_val = torch.nonzero(cat_tran_neighbor_val).size(0)
        cat_poi_relation = torch.sum(cat_poi_relationship[cat_tran_neighbor_idx], dim=-2)  # (C,P)
        poi_rela_neighbor = cat_poi_relation[poi2cat[:, 1]]  # (P,P)
        num_poi_rela_neighbor = torch.nonzero(poi_rela_neighbor).size(0)  # (C,10)2452688029,val<23 7248224060
        # %%==========
        poi_distance = torch.zeros([poi_count, poi_count], dtype=torch.float32)
        distance = torch.exp(-dis_poi_dis).float()
        distance[dis_poi_dis > args.tran_dis_threshold] = 0  # 4sq:-1,gowalla:0.5
        poi_distance[torch.arange(poi_count).unsqueeze(-1), dis_poi_idx] = distance
        poi_distance[poi_rela_neighbor == 0] = 0
        mun_poi_distance = torch.nonzero(poi_distance).size(0)
        # %% ===============================
        poi_rela_neighbor_val, poi_rela_neighbor_idx = poi_distance.topk(args.topk_poi_transition, largest=True)
        poi_rela_neighbor_idx[poi_rela_neighbor_val == 0] = -1
        poi_rela_neighbor_idx = deduplicate_matrix(poi_rela_neighbor_idx, poi_tran_neighbor_idx)
        poi_rela_neighbor_idx = torch.flip(poi_rela_neighbor_idx, dims=[-1])
        # %% ===============================
        mask1 = (poi_tran_neighbor_idx != -1)
        mask2 = (poi_rela_neighbor_idx != -1)
        poi_tran_neighbor_mask = torch.where(mask1 + mask2, 0, float('-inf'))

        poi_rela_neighbor_idx[poi_tran_neighbor_idx != -1] = 0
        poi_tran_neighbor_idx[poi_tran_neighbor_idx == -1] = 0
        poi_rela_neighbor_idx[poi_rela_neighbor_idx == -1] = 0
        num_poi_tran_neighbor_mask = torch.sum(mask1 + mask2)
        poi_tran_neighbor = poi_tran_neighbor_idx + poi_rela_neighbor_idx
        # ======================================
        # gowalla:(20,0.5)1197508,(20,2.5)1602195,(30,2.5)2085598 (40,2.5)2461734 (50,2.5)2741235
        # gowalla2:(30,3.5)2270230 (40,5.5)3001839 (50,7.6)3760304
        # #4sq2:(30,2.5)1470617,(40,3.0)1869735,(50,4.0)2304186
    else:
        mask = (poi_tran_neighbor_val != 0)
        poi_tran_neighbor_mask = torch.where(mask, 0, float('-inf'))
        poi_tran_neighbor_idx[poi_tran_neighbor_val == 0] = 0
        poi_tran_neighbor = poi_tran_neighbor_idx

    poi_tran_neighbor_mask = coo_matrix(poi_tran_neighbor_mask)
    poi_transition = coo_matrix(poi_transition)  # (P,P)

    tran_neighbor_data = {
        'poi_transition': poi_transition,
        'poi_tran_neighbor': poi_tran_neighbor,
        'poi_tran_neighbor_mask': poi_tran_neighbor_mask}
    torch.save(tran_neighbor_data, osp.join(data_path, 'tran_neighbor',
                                            'topk_poi_transition_' + str(args.topk_poi_transition) +
                                            'tran_dis_' + str(args.tran_dis_threshold) + '.pt'))


def build_poi_bias_data():
    # # %% =============================================================================================================
    tran_neighbor_data = torch.load(osp.join(data_path, 'tran_neighbor',
                                             'topk_poi_transition_' + str(args.topk_poi_transition) +
                                             'tran_dis_' + str(args.tran_dis_threshold) + '.pt'))

    poi_transition = tran_neighbor_data.get('poi_transition')
    poi_tran_neighbor = tran_neighbor_data.get('poi_tran_neighbor')
    poi_tran_neighbor_mask = torch.tensor(coo_matrix.toarray(tran_neighbor_data.get('poi_tran_neighbor_mask')))

    poi_transition = sparse_matrix_to_tensor(poi_transition)
    print('normalized poi transition bias2')
    poi_transition2 = torch.sparse.mm(poi_transition, poi_transition)
    print('normalized poi transition bias3')
    poi_transition3 = torch.sparse.mm(poi_transition2, poi_transition)
    print('normalized poi transition bias4')
    poi_transition4 = torch.sparse.mm(poi_transition3, poi_transition)
    print('generate poi transition bias')
    # (p,p,4)
    poi_transition_bias = torch.stack(
        [poi_transition.to_dense()[torch.arange(poi_count).unsqueeze(-1), poi_tran_neighbor],
         poi_transition2.to_dense()[torch.arange(poi_count).unsqueeze(-1), poi_tran_neighbor],
         poi_transition3.to_dense()[torch.arange(poi_count).unsqueeze(-1), poi_tran_neighbor],
         poi_transition4.to_dense()[torch.arange(poi_count).unsqueeze(-1), poi_tran_neighbor]], dim=-1)
    poi_transition_bias[poi_tran_neighbor_mask != 0] = 0  # (P,20,4)
    poi_bias_data = {
        'poi_transition_bias': poi_transition_bias}
    torch.save(poi_bias_data,
               osp.join(data_path, 'poi_bias', 'poi_bias_' +
                        'topk_poi_transition_' + str(args.topk_poi_transition) +
                        'tran_dis_' + str(args.tran_dis_threshold) + '.pt'))


if __name__ == '__main__':
    df = pd.read_csv(osp.join('data', args.dataset_name, 'sample.csv'))
    df_pois = pd.read_csv(osp.join('data', args.dataset_name, 'poi_sample.csv'))
    dis_poi_idx = torch.tensor(pd.read_csv(osp.join('data', args.dataset_name, 'nearest_poi_idx.csv')).values)
    dis_poi_dis = torch.tensor(pd.read_csv(osp.join('data', args.dataset_name, 'nearest_poi_dis.csv')).values)

    print(f'preprocess:{args.dataset_name}')
    time_start = time.time()
    tra_user_id = []
    tra_poi_id = []
    tra_cat_id = []

    user_list = set(df['UserId'].to_list())
    poi_list = df_pois['PoiId'].to_list()
    cat_list = set(df_pois['CatId'].to_list())

    user_count = len(user_list)
    poi_count = len(poi_list)
    cat_count = len(cat_list)
    #
    poi2cat = torch.tensor(df_pois[['PoiId', 'CatId']].values)
    poi2coord = torch.tensor(df_pois[['Latitude', 'Longitude']].values)

    grouped = df.groupby('UserId')
    for user_id, traj in tqdm(grouped, desc='load trajectory'):
        tra_user_id.append(traj['UserId'].tolist())
        tra_poi_id.append(traj['PoiId'].tolist())
        tra_cat_id.append(traj['CatId'].tolist())
    data_path = osp.join('graph', args.dataset_name)
    if not osp.isdir(data_path):
        os.makedirs(data_path)
    # %% ===============================================================================================================

    build_user_data()
    # build_cat_data()
    # build_dis_threshold_neighbor_data()
    # build_tran_neighbor_data()

    build_poi_bias_data()
