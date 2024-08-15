from torch.utils.data import Dataset
from setting import SetParameter
import torch
from tqdm import tqdm

args = SetParameter()


class TraDataset(Dataset):

    def __init__(self, tra_user_id, tra_poi_id, tra_cat_id, tra_time_slot, data_tag):

        self.all_user = []

        self.all_poi = []
        self.all_cat = []
        self.all_time_slot = []

        self.lbl_poi = []
        self.lbl_cat = []
        self.lbl_time_slot = []

        for i in tqdm(range(len(tra_user_id)), desc='split label data'):
            train_idx = int((len(tra_poi_id[i]) - 1) * 0.8)
            if data_tag == 'train':
                idx_start = 0
                idx_end = train_idx
                lbl_idx_start = 1
                lbl_idx_end = train_idx + 1
            else:
                idx_start = train_idx
                idx_end = -1
                lbl_idx_start = train_idx + 1
                lbl_idx_end = None

            self.all_user.append(tra_user_id[i][idx_start:idx_end])

            self.all_poi.append(tra_poi_id[i][idx_start:idx_end])
            self.all_cat.append(tra_cat_id[i][idx_start:idx_end])
            self.all_time_slot.append(tra_time_slot[i][idx_start:idx_end])

            self.lbl_poi.append(tra_poi_id[i][lbl_idx_start:lbl_idx_end])
            self.lbl_cat.append(tra_cat_id[i][lbl_idx_start:lbl_idx_end])
            self.lbl_time_slot.append(tra_time_slot[i][lbl_idx_start:lbl_idx_end])
        self.sub_user_id = []

        self.sub_poi_id = []
        self.sub_cat_id = []
        self.sub_time_slot = []

        self.lbl_sub_poi_id = []
        self.lbl_sub_cat_id = []
        self.lbl_sub_time_slot = []

        for i in tqdm(range(len(tra_user_id)), desc=f'split {data_tag} dataset'):
            seq_count = len(self.all_poi[i]) // args.avg_sub_length
            assert seq_count > 0, 'fix seq-length and min-checkins in order to have at least one test sequence in a 80/20 split!'
            for j in range(seq_count):
                start = j * args.avg_sub_length
                end = (j + 1) * args.avg_sub_length
                self.sub_user_id.append(self.all_user[i][start:end])

                self.sub_poi_id.append(self.all_poi[i][start:end])
                self.sub_cat_id.append(self.all_cat[i][start:end])
                self.sub_time_slot.append(self.all_time_slot[i][start:end])

                self.lbl_sub_poi_id.append(self.lbl_poi[i][start:end])
                self.lbl_sub_cat_id.append(self.lbl_cat[i][start:end])
                self.lbl_sub_time_slot.append(self.lbl_time_slot[i][start:end])

        self.sub_user_id = torch.tensor(self.sub_user_id)

        self.sub_poi_id = torch.tensor(self.sub_poi_id)
        self.sub_cat_id = torch.tensor(self.sub_cat_id)
        self.sub_time_slot = torch.tensor(self.sub_time_slot)

        self.lbl_sub_poi_id = torch.tensor(self.lbl_sub_poi_id)
        self.lbl_sub_cat_id = torch.tensor(self.lbl_sub_cat_id)
        self.lbl_sub_time_slot = torch.tensor(self.lbl_sub_time_slot)

    def __getitem__(self, idx):
        return self.sub_user_id[idx], \
               self.sub_poi_id[idx], self.lbl_sub_poi_id[idx], self.sub_cat_id[idx], self.lbl_sub_cat_id[idx], \
               self.sub_time_slot[idx], self.lbl_sub_time_slot[idx]

    def __len__(self):
        return len(self.sub_poi_id)
