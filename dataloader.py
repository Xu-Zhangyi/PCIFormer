from dataset import TraDataset
from setting import SetParameter
from tqdm import tqdm
import torch
import pandas as pd
from utils import calculate_time_slot
import os.path as osp

args = SetParameter()


# U 是用户计数（user count），P 是地点计数（poi count），C 是类别计数（category count）
class TraDataloader:

    def __init__(self, sample, poi_sample, logger):
        self.tra_user_id = []
        self.tra_poi_id = []
        self.tra_cat_id = []
        self.tra_time_slot = []

        # %% ===========================================================================================================
        poi_sample = pd.read_csv(poi_sample)
        df = pd.read_csv(sample)

        self.user_list = set(df['UserId'].to_list())
        self.poi_list = poi_sample['PoiId'].to_list()
        self.cat_list = set(poi_sample['CatId'].to_list())

        self.user_count = len(self.user_list)
        self.poi_count = len(self.poi_list)
        self.cat_count = len(self.cat_list)

        # %% ===========================================================================================================
        data_path = osp.join('graph', args.dataset_name)
        #
        tran_neighbor_path = osp.join(data_path, 'tran_neighbor','topk_poi_transition_' + str(args.topk_poi_transition) +'tran_dis2_' + str(args.tran_dis_threshold) + '.pt')
        tran_neighbor_data = torch.load(tran_neighbor_path)
        #
        dis_neighbor_path = osp.join(data_path, 'dis_neighbor','dis_threshold_' + str(args.dis_threshold) + 'topk_poi_distance_' + str(args.topk_poi_distance) + '.pt')
        dis_neighbor_data = torch.load(dis_neighbor_path)
        #
        user_path = osp.join(data_path, 'user', 'topk_user_poi_pref_' + str(args.topk_user_poi_pref) + '.pt')
        user_data = torch.load(user_path)
        #
        cat_path = osp.join(data_path, 'cat', 'cat.pt')
        cat_data = torch.load(cat_path)
        #
        poi_bias_path = osp.join(data_path, 'poi_bias', 'poi_bias_topk_poi_transition_' + str(args.topk_poi_transition) +'tran_dis2_' + str(args.tran_dis_threshold) + '.pt')
        poi_bias_data = torch.load(poi_bias_path)
        #
        logger.info(f'tran neighbor path:{tran_neighbor_path}')
        logger.info(f'dis neighbor path:{dis_neighbor_path}')
        logger.info(f'cat path:{cat_path}')
        logger.info(f'user path:{user_path}')
        logger.info(f'poi bias path:{poi_bias_path}')
        self.poi_transition = tran_neighbor_data.get('poi_transition')
        self.poi_tran_neighbor = tran_neighbor_data.get('poi_tran_neighbor')
        self.poi_tran_neighbor_mask = tran_neighbor_data.get('poi_tran_neighbor_mask')

        self.poi_distance = dis_neighbor_data.get('poi_distance')
        self.poi_dis_neighbor = dis_neighbor_data.get('poi_dis_neighbor')
        self.poi_dis_neighbor_mask = dis_neighbor_data.get('poi_dis_neighbor_mask')

        self.cat_transition = cat_data.get('cat_transition')
        self.cat_poi_mask = cat_data.get('cat_poi_mask')

        self.user_poi_pref_idx = user_data.get('user_poi_pref_idx')
        self.user_poi_pref_val = user_data.get('user_poi_pref_val')
        self.user_poi_pref_mask = user_data.get('user_poi_pref_mask')

        self.poi_transition_bias = poi_bias_data.get('poi_transition_bias')

        grouped = df.groupby('UserId')
        for user_id, traj in tqdm(grouped, desc='load trajectory'):
            self.tra_user_id.append(traj['UserId'].tolist())
            self.tra_poi_id.append(traj['PoiId'].tolist())
            self.tra_cat_id.append(traj['CatId'].tolist())
            self.tra_time_slot.append(traj['UTCTime'].apply(calculate_time_slot).tolist())

        # %% ===========================================================================================================

    def create_dataset(self, data_tag):
        return TraDataset(self.tra_user_id, self.tra_poi_id, self.tra_cat_id, self.tra_time_slot, data_tag)
