#   PCIFormer
PCIFormer是一个基于上下文图增强的人类移动预测模型。它根据用户的历史访问记录，为其提供下一步行动建议。

## 特性
* 同时考虑位置之间的各种关系
* 建立位置与类别以及用户之间的关联
* 多任务环境中进行预测
## 致谢
本项目受科技部重点研发计划项目支持（2022YFE0137800）
## 运行方法:
Source codes for LCIFormer
1. Download the Gowalla dataset and the Foursquare dataset. The URLs of datasets have been listed.
2. Run 'preprocess_gowalla_4sq.py' to obtain the initial dataset for training and test.
3. Run 'build_dataset_graph.py' to get location graph information.
4. Run 'train.py' to train PCIFormer.
