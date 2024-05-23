import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import time
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import h5py
import cv2
import shutil
from model import CSRNet
import matplotlib.pyplot as plt
import json
import datetime


def save_checkpoint(state, is_best, task_id, save_dir="./model/", filename="checkpoint.pth.tar"):
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)


def load_data(rgb_path, tir_path, gt_path, train=True):
    rgb_img = Image.open(rgb_path).convert('RGB')
    tir_img = Image.open(tir_path).convert('RGB')  # 假设红外图像也是3通道
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(target, 
                        (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return rgb_img, tir_img, target

class ImgDataset(Dataset):
    def __init__(self, rgb_dir, tir_dir, gt_dir, shape=None, shuffle=True, transform_rgb=None,transform_tir=None, train=False, batch_size=1, num_workers=4):
        self.rgb_dir = rgb_dir
        self.tir_dir = tir_dir
        self.gt_dir = gt_dir
        self.transform_rgb = transform_rgb
        self.transform_tir = transform_tir
        self.train = train

        self.img_paths = [os.path.join(rgb_dir, filename) for filename in os.listdir(rgb_dir) if filename.endswith('.jpg')]
        self.tir_paths = [os.path.join(tir_dir, filename) for filename in os.listdir(tir_dir) if filename.endswith('.jpg')]
        self.gt_paths = [os.path.join(gt_dir, filename.replace('.jpg', '.h5')) for filename in os.listdir(rgb_dir) if filename.endswith('.jpg')]
        
        if shuffle:
            combined = list(zip(self.img_paths, self.tir_paths))
            random.shuffle(combined)
            self.img_paths, self.tir_paths = zip(*combined)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        tir_path = self.tir_paths[index]
        img_name = os.path.basename(img_path)
        gt_name = os.path.splitext(img_name)[0] + '.h5'
        gt_path = os.path.join(self.gt_dir, gt_name)

        rgb_img, tir_img, target = load_data(
            img_path, tir_path, gt_path, self.train)

        if self.transform_rgb is not None:
            rgb_img = self.transform_rgb(rgb_img)
        if self.transform_tir is not None:
            tir_img = self.transform_tir(tir_img)

        return rgb_img, tir_img, target

# 参数设置
lr = 1e-7
original_lr = lr
batch_size = 1  # 修改为4，加快训练
momentum = 0.95
decay = 5*1e-4
epochs = 400
# TODO2:scales是否需要调整？
steps = [-1, 1, 120, 250, 300, 350]
scales = [1, 1, 1, 0.5, 0.2, 0.1]   
workers = 4
seed = time.time()
print_freq = 30
img_dir = "./dataset/train/rgb/"
# 新增红外图像路径 Thermal Imaging Radiometer
tir_dir = "./dataset/train/tir/"
gt_dir = "./dataset/train/hdf5s/"
# 训练中断时，可以从中间继续
pre = None
task = ""
rgb_pre = "model/rgb-model_best.pth.tar"
tir_pre = "model/tir-model_best.pth.tar"
weight_pre = "model/best_weights.pth"

def main():
    start_epoch = 0
    best_prec1 = 1e6

    torch.cuda.manual_seed(seed)

    # 定义两个模型
    model_rgb = CSRNet()
    model_tir = CSRNet()

    model_rgb = model_rgb.cuda()
    model_tir = model_tir.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer_rgb = torch.optim.SGD(model_rgb.parameters(), lr, momentum=momentum, weight_decay=decay)
    optimizer_tir = torch.optim.SGD(model_tir.parameters(), lr, momentum=momentum, weight_decay=decay)

    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_tir = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImgDataset(
        img_dir, tir_dir, gt_dir, transform_rgb=transform_rgb, transform_tir=transform_tir, train=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # if pre:
    #     if os.path.isfile(pre):
    #         print("=> loading checkpoint '{}'".format(pre))
    #         checkpoint = torch.load(pre)
    #         start_epoch = checkpoint['epoch']
    #         best_prec1 = checkpoint['best_prec1']
    #         model_rgb.load_state_dict(checkpoint['state_dict_rgb'])
    #         optimizer_rgb.load_state_dict(checkpoint['optimizer_rgb'])
    #         model_tir.load_state_dict(checkpoint['state_dict_tir'])
    #         optimizer_tir.load_state_dict(checkpoint['optimizer_tir'])
    #         print("=> loaded checkpoint '{}' (epoch {})".format(pre, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(pre))
    if rgb_pre:
        if os.path.isfile(rgb_pre):
            print("=> loading checkpoint '{}'".format(rgb_pre))
            checkpoint = torch.load(rgb_pre)
            start_epoch = checkpoint['epoch']
            model_rgb.load_state_dict(checkpoint['state_dict'])
            optimizer_rgb.load_state_dict(checkpoint['optimizer'])
            best_prec1 = checkpoint['best_prec1']
            print("=> loaded checkpoint '{}' (epoch {})".format(rgb_pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(rgb_pre))

    if tir_pre:
        if os.path.isfile(tir_pre):
            print("=> loading checkpoint '{}'".format(tir_pre))
            checkpoint = torch.load(tir_pre)
            model_tir.load_state_dict(checkpoint['state_dict'])
            optimizer_tir.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(tir_pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(tir_pre))

    MAEs = {}
    alpha = torch.tensor(0.5, requires_grad=True, device='cuda')
    beta = torch.tensor(0.5, requires_grad=True, device='cuda')
    optimizer_alpha_beta = torch.optim.SGD([alpha, beta], lr=lr)

    if weight_pre:
        if os.path.isfile(weight_pre):
            print("=> loading checkpoint '{}'".format(weight_pre))
            checkpoint = torch.load(weight_pre)
            alpha = torch.tensor(checkpoint['alpha'], device='cuda')
            beta = torch.tensor(checkpoint['beta'], device='cuda')
            print("=> loaded checkpoint '{}'".format(weight_pre))
        else:
            print("=> no checkpoint found at '{}'".format(weight_pre))

    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer_rgb, epoch)
        adjust_learning_rate(optimizer_tir, epoch)

        train(model_rgb, model_tir, alpha, beta, criterion, optimizer_rgb, optimizer_tir, optimizer_alpha_beta, epoch, train_loader)
        prec1 = validate(model_rgb, model_tir, val_loader, alpha, beta)

        MAEs[epoch] = prec1.item()  # Convert Tensor to float

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model_rgb.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer_rgb.state_dict(),
        }, is_best, "rgb-")

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model_tir.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer_tir.state_dict(),
        }, is_best, "tir-")
        if epoch % 50 == 0:
            with open('./pic/decision-fusion-MAEs.json', 'w') as f:
                json.dump(MAEs, f, indent=4)

        # Save the best alpha and beta
        if is_best:
            torch.save({'alpha': alpha.item(), 'beta': beta.item()}, './model/best_weights.pth')
    
    with open('./pic/decision-fusion-MAEs.json', 'w') as f:
        json.dump(MAEs, f, indent=4)

    # 绘制MAEs曲线
    # 确保在使用 numpy 之前将张量移动到 CPU
    epochs_list_cpu = torch.tensor(list(MAEs.keys())).cpu().numpy()
    maes_list_cpu = torch.tensor(list(MAEs.values())).cpu().numpy()

    plt.plot(epochs_list_cpu, maes_list_cpu)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE of each epoch')

    if not os.path.exists('pic'):
        os.makedirs('pic')

    plt.savefig('pic/decision-fusion-MAEs.png')


    
def train(model_rgb, model_tir, alpha, beta, criterion, optimizer_rgb, optimizer_tir, optimizer_alpha_beta, epoch, train_loader):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), lr))

    model_rgb.train()
    model_tir.train()

    end = time.time()

    for i, (img_rgb, img_tir, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img_rgb = img_rgb.cuda()
        img_tir = img_tir.cuda()
        img_rgb = Variable(img_rgb)
        img_tir = Variable(img_tir)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)

        output_rgb = model_rgb(img_rgb)
        output_tir = model_tir(img_tir)

        # 决策融合，使用可学习的权重进行加权和
        output = alpha * output_rgb + beta * output_tir

        loss = criterion(output, target)

        losses.update(loss.item(), img_rgb.size(0))
        optimizer_rgb.zero_grad()
        optimizer_tir.zero_grad()
        optimizer_alpha_beta.zero_grad()
        loss.backward()
        optimizer_rgb.step()
        optimizer_tir.step()
        optimizer_alpha_beta.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses))


def validate(model_rgb, model_tir, val_loader, alpha, beta):
    print('begin test')

    model_rgb.eval()
    model_tir.eval()
    mae = 0

    with torch.no_grad():
        for i, (img_rgb, img_tir, target) in enumerate(val_loader):
            img_rgb = img_rgb.cuda()
            img_tir = img_tir.cuda()
            img_rgb = Variable(img_rgb)
            img_tir = Variable(img_tir)
            target = target.type(torch.FloatTensor).unsqueeze(1).cuda()

            output_rgb = model_rgb(img_rgb)
            output_tir = model_tir(img_tir)

            # 决策融合，使用训练好的权重进行加权和
            output = alpha * output_rgb + beta * output_tir

            mae += abs(output.data.sum() - target.sum().type(torch.FloatTensor).cuda())

        mae = mae / len(val_loader)
        print(' * MAE {mae:.3f} '.format(mae=mae))

    return mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""

    lr = original_lr

    for i in range(len(steps)):

        scale = scales[i] if i < len(scales) else 1

        if epoch >= steps[i]:
            lr = lr * scale
            if epoch == steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
