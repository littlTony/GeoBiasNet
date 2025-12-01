import os
import sys
import time
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import h36motion3d as datasets
from utils import util, log
from utils.opt import Options
from model.model_h36m_long import MultiStageModel_4 as BARNet
import random

# 设置环境变量解决CUDA错误
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义动作常量
ACTIONS = [
    "walking", "eating", "smoking", "discussion", "directions",
    "greeting", "phoning", "posing", "purchases", "sitting",
    "sittingdown", "takingphoto", "waiting", "walkingdog",
    "walkingtogether"
]

# 定义使用的维度索引
DIM_USED = np.array([
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92
])

# 定义需要忽略的关节和对应的索引
JOINT_TO_IGNORE = np.array([16, 20, 23, 24, 28, 31])
INDEX_TO_IGNORE = np.concatenate((
    JOINT_TO_IGNORE * 3, 
    JOINT_TO_IGNORE * 3 + 1, 
    JOINT_TO_IGNORE * 3 + 2
))

# 定义对称关节和对应的索引
JOINT_EQUAL = np.array([13, 19, 22, 13, 27, 30])
INDEX_TO_EQUAL = np.concatenate((
    JOINT_EQUAL * 3, 
    JOINT_EQUAL * 3 + 1, 
    JOINT_EQUAL * 3 + 2
))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def eval(opt):
    """模型评估函数"""
    print('>>> 创建模型')
    net_pred = BARNet(opt=opt)
    net_pred.to(opt.cuda_idx)
    net_pred.eval()

    # 加载模型
    model_path = 'checkpoint/main_h36m_ddp_long_all_in10_out50_ks10_dctn60_dropout_0.3_lr_0.005_d_model_16_e_6_d_6/ckpt_best.pth.tar'
    print(f">>> 从 '{model_path}' 加载检查点")
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    net_pred.load_state_dict(ckpt['state_dict'])
    print(f">>> 检查点加载完成 (epoch: {ckpt['epoch']} | err: {ckpt['err']})")

    # 准备数据加载器
    data_loader = {}
    for action in ACTIONS:
        dataset = datasets.Datasets(opt=opt, split=2, actions=action)
        data_loader[action] = DataLoader(
            dataset, 
            batch_size=opt.test_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=True
        )

    # 执行评估
    is_create = True
    avg_ret_log = []
    for action in ACTIONS:
        ret_test = run_model(net_pred, is_train=3, data_loader=data_loader[action], opt=opt, action=action)
        
        # 准备日志数据
        ret_log = [action]
        head = ['action']
        for key in ret_test.keys():
            ret_log.append(ret_test[key])
            head.append(f'test_{key}')
            
        avg_ret_log.append(ret_log[1:])
        log.save_csv_eval_log(opt, head, ret_log, is_create=is_create)
        is_create = False

    # 保存平均结果
    avg_ret_log = np.array(avg_ret_log, dtype=np.float64).mean(axis=0)
    write_ret_log = ret_log.copy()
    write_ret_log[0] = 'avg'
    write_ret_log[1:] = avg_ret_log
    log.save_csv_eval_log(opt, head, write_ret_log, is_create=False)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=None, opt=None, action=None):
    """
    运行模型训练/评估
    is_train: 0-训练, 1-验证, 3-测试 
    """
    # 设置模型模式
    net_pred.train() if is_train == 0 else net_pred.eval()

    # 初始化变量
    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = (np.arange(opt.output_n) + 1) * 40
        m_p3d_h36 = np.zeros(opt.output_n)
        
    n = 0
    in_n = opt.input_n
    out_n = opt.train_output_n
    seq_in = opt.kernel_size
    start_time = time.time()

    # 遍历数据加载器
    for i, (p3d_h36) in enumerate(data_loader):
        batch_size, seq_n, all_dim = p3d_h36.shape
        # 跳过单样本批次（训练时）
        if batch_size == 1 and is_train == 0:
            continue
            
        n += batch_size
        batch_time = time.time()
        p3d_h36 = p3d_h36.float().to(opt.cuda_idx)

        # 准备输入数据
        input_data = p3d_h36[:, :, DIM_USED].clone()

        # 前向传播
        outputs = net_pred(input_data)
        p3d_out_all_fin, _, _, _, _, = outputs

        total_pre = p3d_out_all_fin[:, seq_in:]

        # 处理输出
        out_n = opt.output_n
        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, DIM_USED] = total_pre
        p3d_out[:, :, INDEX_TO_IGNORE] = p3d_out[:, :, INDEX_TO_EQUAL]
        p3d_out = p3d_out.reshape([-1, out_n, all_dim // 3, 3])
        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, all_dim // 3, 3])

        # 计算每个时间步的位置误差
        mpjpe = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
        m_p3d_h36 += mpjpe.cpu().data.numpy()

        # 打印进度
        if i % 1000 == 0:
            elapsed = time.time() - start_time
            print(f'{i+1}/{len(data_loader)} | '
                  f'批次时间: {time.time() - batch_time:.3f}s | '
                  f'总时间: {elapsed:.0f}s')

    # 准备返回结果
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n
        
    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 /= n
        for j in range(out_n):
            ret[f"#{titles[j]}ms"] = m_p3d_h36[j]
            
    return ret


if __name__ == '__main__':
    # 初始化配置和全局变量
    option = Options().parse()

    set_seed(334)

    # 执行主程序
    eval(option)
