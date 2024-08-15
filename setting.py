import torch


# 参数设置
class SetParameter:
    def __init__(self):
        self.dataset_name = 'gowalla'
        self.min_user_freq = 101
        self.avg_sub_length = 20

        self.epochs = 100
        self.test_epoch = 0

        self.emb_dim = 128

        self.poi_gcn_dim = [128, 256, 128]
        self.cat_gcn_dim = [128, 128]

        self.num_heads = 2
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        self.num_layers = 1
        self.num_seq_layers = 1

        # 其他设置
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        if self.dataset_name == 'gowalla':
            self.dis_threshold = 0.5

            self.topk_poi_distance = 15
            self.topk_poi_transition = 30

            self.topk_user_poi_pref = 80

            self.train_batch_size = 200
            self.test_batch_size = 200

            self.tran_dis_threshold = 3.5
            self.milestones = [15, 20, 25, 30, 35, 55, 65, 75]
        elif self.dataset_name == '4sq':
            self.dis_threshold = 1.5

            self.topk_poi_distance = 15
            self.topk_poi_transition = 30

            self.topk_user_poi_pref = 40

            self.train_batch_size = 1000
            self.test_batch_size = 400

            self.tran_dis_threshold = 2.5
            self.milestones =  [25, 50, 75, 80, 90, 100, 110]

    def __str__(self):
        return '\n'.join([f'{key}: {value}' for key, value in vars(self).items()])
