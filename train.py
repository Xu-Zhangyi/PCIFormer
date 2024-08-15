import time
import numpy as np
import torch.nn as nn
from scipy.sparse import coo_matrix
from setting import SetParameter
from dataloader import TraDataloader
from torch.utils.data import DataLoader
from utils import Logger, generate_src_mask, count_parameters
from utils import zipdir, set_seed
from tqdm import tqdm
import torch
from evaluation import eval_every_timestep
from network import TransformerModel
import os
import datetime
import os.path as osp
import zipfile
import pathlib

set_seed(11)
args = SetParameter()

current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = osp.join('log', args.dataset_name, current_time)
if not osp.isdir(log_path):
    os.makedirs(log_path)
logger = Logger(log_path).print_log()
# Save python code
zipf = zipfile.ZipFile(os.path.join(log_path, 'code.zip'), 'w', zipfile.ZIP_DEFLATED)
zipdir(pathlib.Path().absolute(), zipf, include_format=['.py'])
zipf.close()

#  ===================================================================================================================
tra_dataloader = TraDataloader('data/' + args.dataset_name + '/sample.csv',
                               'data/' + args.dataset_name + '/poi_sample.csv',logger)

user_count = tra_dataloader.user_count
poi_count = tra_dataloader.poi_count
cat_count = tra_dataloader.cat_count

poi_transition = tra_dataloader.poi_transition
poi_tran_neighbor = tra_dataloader.poi_tran_neighbor.to(args.device)
poi_tran_neighbor_mask = torch.tensor(coo_matrix.toarray(tra_dataloader.poi_tran_neighbor_mask)).to(args.device)

poi_distance = tra_dataloader.poi_distance
poi_dis_neighbor = tra_dataloader.poi_dis_neighbor.to(args.device)
poi_dis_neighbor_mask = torch.tensor(coo_matrix.toarray(tra_dataloader.poi_dis_neighbor_mask)).to(args.device)

cat_transition = tra_dataloader.cat_transition
cat_poi_mask = torch.tensor(coo_matrix.toarray(tra_dataloader.cat_poi_mask)).repeat(
    args.num_heads, 1, 1).to(args.device)

user_poi_pref_idx = tra_dataloader.user_poi_pref_idx.to(args.device)
user_poi_pref_val = tra_dataloader.user_poi_pref_val.to(args.device)
user_poi_pref_mask = tra_dataloader.user_poi_pref_mask.to(args.device)

poi_transition_bias = tra_dataloader.poi_transition_bias.to(args.device)

train_dataset = tra_dataloader.create_dataset('train')  # 训练数据
test_dataset = tra_dataloader.create_dataset('test')  # 测试数据

train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, drop_last=False)

#  ====================================================================================================================
transformer_model = TransformerModel(user_count=user_count, poi_count=poi_count, cat_count=cat_count,
                                     poi_transition=poi_transition,
                                     poi_tran_neighbor=poi_tran_neighbor,
                                     poi_tran_neighbor_mask=poi_tran_neighbor_mask,

                                     poi_distance=poi_distance,
                                     poi_dis_neighbor=poi_dis_neighbor,
                                     poi_dis_neighbor_mask=poi_dis_neighbor_mask,

                                     poi_transition_bias=poi_transition_bias,

                                     cat_transition=cat_transition,
                                     cat_poi_mask=cat_poi_mask,

                                     user_poi_pref_idx=user_poi_pref_idx,
                                     user_poi_pref_val=user_poi_pref_val,
                                     user_poi_pref_mask=user_poi_pref_mask,
                                     ).to(args.device)

optimizer = torch.optim.Adam(transformer_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.2)

cross_entropy_loss = nn.CrossEntropyLoss()
cross_entropy_loss_cat = nn.CrossEntropyLoss()
num_parameters = count_parameters(transformer_model)

logger.info(transformer_model)
logger.info(f'Parameters:{num_parameters}')
logger.info(f'user count:{user_count}')
logger.info(f'poi count:{poi_count}')
logger.info(f'cat count:{cat_count}')
logger.info(f'train tra count:{len(train_dataset.sub_user_id)}')
logger.info(f'test tra count:{len(test_dataset.sub_user_id)}')
logger.info(args)
logger.info(f'successfully load dataset')
# ======================================================================================================================
best_metrics = 0
for epoch in tqdm(range(args.epochs), desc='Training'):
    logger.info(f'~~~ Train (Epoch: {epoch}) ~~~')
    losses = []
    loss_poi_s = []
    loss_cat_s = []

    epoch_start = time.time()

    for i, (user_id, x_poi_id, y_poi_id, x_cat_id, y_cat_id, x_time_slot, y_time_slot) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        tra_mask = generate_src_mask(len(user_id), args.num_heads, len(x_poi_id[0])).to(args.device)
        transformer_model.train()
        y_pre, y_cat_pre = transformer_model(user_id, x_poi_id, x_cat_id, x_time_slot, attn_mask=tra_mask)

        y_pre = y_pre.reshape(-1, poi_count)
        y_poi_id = y_poi_id.reshape(-1).to(args.device)
        loss_poi = cross_entropy_loss(y_pre, y_poi_id)

        y_cat_pre = y_cat_pre.reshape(-1, cat_count)
        y_cat_id = y_cat_id.reshape(-1).to(args.device)
        loss_cat = cross_entropy_loss_cat(y_cat_pre, y_cat_id)

        loss = loss_poi + loss_cat
        loss.backward()

        losses.append(loss.item())
        loss_poi_s.append(loss_poi.item())
        loss_cat_s.append(loss_cat.item())

        optimizer.step()

    epoch_end = time.time()
    scheduler.step()
    epoch_loss = np.mean(losses)
    logger.info('One training need {:.2f}s'.format(epoch_end - epoch_start))
    logger.info(f'Epoch:{epoch}/{args.epochs}')
    logger.info(f'Used learning rate:{scheduler.get_last_lr()[0]}')
    logger.info(f'Avg Loss:{epoch_loss}')
    logger.info(f'Avg Loss poi:{np.mean(loss_poi_s)}')
    logger.info(f'Avg Loss cat:{np.mean(loss_cat_s)}')

    # ===============================================================================================================
    if (epoch) >= args.test_epoch:
        logger.info(f'~~~ Test Set Evaluation (Epoch: {epoch}) ~~~')

        count_time_step = 0
        a_t_1 = []
        a_t_5 = []
        a_t_10 = []
        a_t_20 = []
        m = []

        count_time_step_cat = 0
        a_t_1_cat = []
        a_t_5_cat = []
        a_t_10_cat = []
        a_t_20_cat = []
        m_cat = []

        epoch_start = time.time()
        with torch.no_grad():
            for i, (user_id,
                    x_poi_id, y_poi_id, x_cat_id, y_cat_id, x_time_slot, y_time_slot) in tqdm(enumerate(test_loader)):
                tra_mask = generate_src_mask(len(user_id), args.num_heads, len(x_poi_id[0])).to(args.device)
                transformer_model.eval()
                y_pre, y_cat_pre = transformer_model(user_id, x_poi_id, x_cat_id, x_time_slot, attn_mask=tra_mask)
                # %========================================
                y_pre = y_pre.reshape(-1, poi_count)
                y_poi_id = y_poi_id.reshape(-1).to(args.device)
                acc_top_1, acc_top_5, acc_top_10, acc_top_20, mrr, l = eval_every_timestep(y_poi_id, y_pre)
                count_time_step += l
                a_t_1.append(acc_top_1)
                a_t_5.append(acc_top_5)
                a_t_10.append(acc_top_10)
                a_t_20.append(acc_top_20)
                m.append(mrr)

                # %========================================
                y_cat_pre = y_cat_pre.reshape(-1, cat_count)
                y_cat_id = y_cat_id.reshape(-1).to(args.device)
                acc_top_1_cat, acc_top_5_cat, acc_top_10_cat, acc_top_20_cat, mrr_cat, l_cat = eval_every_timestep(
                    y_cat_id, y_cat_pre)
                count_time_step_cat += l_cat
                a_t_1_cat.append(acc_top_1_cat)
                a_t_5_cat.append(acc_top_5_cat)
                a_t_10_cat.append(acc_top_10_cat)
                a_t_20_cat.append(acc_top_20_cat)
                m_cat.append(mrr_cat)

            a_t_1 = torch.sum(torch.tensor(a_t_1)) / count_time_step
            a_t_5 = torch.sum(torch.tensor(a_t_5)) / count_time_step
            a_t_10 = torch.sum(torch.tensor(a_t_10)) / count_time_step
            a_t_20 = torch.sum(torch.tensor(a_t_20)) / count_time_step
            m = torch.sum(torch.tensor(m)) / count_time_step
            # %========================================
            a_t_1_cat = torch.sum(torch.tensor(a_t_1_cat)) / count_time_step_cat
            a_t_5_cat = torch.sum(torch.tensor(a_t_5_cat)) / count_time_step_cat
            a_t_10_cat = torch.sum(torch.tensor(a_t_10_cat)) / count_time_step_cat
            a_t_20_cat = torch.sum(torch.tensor(a_t_20_cat)) / count_time_step_cat
            m_cat = torch.sum(torch.tensor(m_cat)) / count_time_step_cat

        epoch_end = time.time()
        logger.info('One test need {:.2f}s'.format(epoch_end - epoch_start))
        logger.info('Test Acc@1 {:.4f}'.format(a_t_1))
        logger.info('Test Acc@5 {:.4f}'.format(a_t_5))
        logger.info('Test Acc@10 {:.4f}'.format(a_t_10))
        logger.info('Test Acc@20 {:.4f}'.format(a_t_20))
        logger.info('Test MRR {:.4f}'.format(m))
        logger.info('~~~')
        logger.info('Test cat Acc@1 {:.4f}'.format(a_t_1_cat))
        logger.info('Test cat Acc@5 {:.4f}'.format(a_t_5_cat))
        logger.info('Test cat Acc@10 {:.4f}'.format(a_t_10_cat))
        logger.info('Test cat Acc@20 {:.4f}'.format(a_t_20_cat))
        logger.info('Test cat MRR {:.4f}'.format(m_cat))

        metrics = a_t_1
        if metrics > best_metrics:
            save_path = os.path.join(log_path, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': transformer_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, save_path)
            best_metrics = metrics
