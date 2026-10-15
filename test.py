import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_msssim import ssim
# from pytorch_msssimm import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from models import DepthNet
from pytorch_ssim import ssim
from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
from utils.metrics import psnr
import time
import cv2

import setproctitle
setproctitle.setproctitle("Dehaze_Test")

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--dehaze_result_dir', default='./results/dehaze_result/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='RESIDE-IN', type=str, help='dataset name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
parser.add_argument('--gpu', default='0', type=str, help='GPUs used for training')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def test_dehaze(test_loader, network, dehaze_result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()
    torch.cuda.empty_cache()
    network.eval()

    os.makedirs(os.path.join(dehaze_result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(dehaze_result_dir, 'dehaze_results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()
        filename = batch['filename'][0]

        with torch.no_grad():
            # 原图输出
            output1 = network(input)[0].clamp_(-1, 1)

          

            # 映射回 [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5

            # 防止溢出，提高 PSNR 精度
            output = output.clamp(0, 1)

            # 更精确 PSNR 计算
            psnr_val = psnr(output, target)

            # 更精确 SSIM 计算
            ssim_val = ssim(output, target).item()

        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test_dehaze: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        write_img(os.path.join(dehaze_result_dir, 'imgs', filename), out_img)

    f_result.close()
    os.rename(os.path.join(dehaze_result_dir, 'dehaze_results.csv'),
              os.path.join(dehaze_result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    target_name = args.model.replace('-', '_')
    model_obj = eval(target_name)

    # 1. 先实例化 DepthNet 作为深度网络参数对象
    DEPTH_NET = DepthNet.DN().cuda()

    # 2. 将 DEPTH_NET 传给 build_net() 实例化主网络
    if hasattr(model_obj, 'build_net'):
        network = model_obj.build_net(depth_net=DEPTH_NET).cuda()
    elif hasattr(model_obj, target_name):
        network = getattr(model_obj, target_name)().cuda()
    elif callable(model_obj):
        network = model_obj().cuda()
    else:
        raise AttributeError(f"在 models 中找不到可调用的 build_net 或 {target_name} 类！")

    # 3. 动态拼接权重路径
    dehaze_ckpt_name = f"{args.model}-dehaze-Indoor.pth"
    depth_ckpt_name = f"{args.model}-depth-Indoor.pth"

    dehaze_model_path = os.path.join(args.save_dir, args.exp, dehaze_ckpt_name)
    depth_model_path = os.path.join(args.save_dir, args.exp, depth_ckpt_name)

    # 4. 加载去雾模型权重
    if not os.path.exists(dehaze_model_path):
        print(f'==> 错误: 找不到去雾权重文件: {dehaze_model_path}')
        exit(0)

    print(f'==> Start testing, current model name: {args.model}')
    print(f'==> Loading dehaze weights: {dehaze_model_path}')
    dehaze_state_dict = torch.load(dehaze_model_path)
    network.load_state_dict(dehaze_state_dict['dehaze_net'])

    if os.path.exists(depth_model_path):
        print(f'==> Loading depth weights: {depth_model_path}')
        depth_state_dict = torch.load(depth_model_path)
        
        # 加载到外部定义的 DEPTH_NET
        DEPTH_NET.load_state_dict(depth_state_dict['depth_net'])
        
        # 同步加载到网络内部的 dpn 属性中
        if hasattr(network, 'dpn'):
            network.dpn.load_state_dict(depth_state_dict['depth_net'])
            print('==> [成功] 已成功将深度权重同步加载到 network.dpn！')
        else:
            print('==> [警告] network 中未找到 dpn 属性！')
    else:
        print(f'==> 提示: 未检测到深度权重文件 ({depth_model_path})，跳过深度加载。')

    # 5. 构建 DataLoader (batch_size=1 保证最稳妥、绝不报错)
    dataset_dir = os.path.join(args.data_dir, args.dataset)
    test_dataset = PairLoader(dataset_dir, 'test', 'test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False
    )

    dehaze_result_dir = os.path.join(args.dehaze_result_dir, args.dataset, args.model)
    test_dehaze(test_loader, network, dehaze_result_dir)