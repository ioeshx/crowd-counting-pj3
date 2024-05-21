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


def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', save_dir='./model/'):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(
            save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)


def load_data(rgb_path, tir_path, gt_path, train=True):
    rgb_img = Image.open(rgb_path).convert('RGB')
    tir_img = Image.open(tir_path).convert('RGB')  # 假设红外图像也是3通道
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(
        target, (target.shape[1] // 8, target.shape[0] // 8), interpolation=cv2.INTER_CUBIC) * 64

    return rgb_img, tir_img, target



class ImgDataset(Dataset):
    def __init__(self, rgb_dir, tir_dir, gt_dir, shape=None, shuffle=True, transform_rgb=None, transform_tir=None,train=False, batch_size=1, num_workers=4):
        self.rgb_dir = rgb_dir
        self.tir_dir = tir_dir
        self.gt_dir = gt_dir
        self.transform_rgb = transform_rgb  # 用于数据增强的transformer
        self.transform_tir = transform_tir
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.rgb_paths = [os.path.join(rgb_dir, filename) for filename in os.listdir(rgb_dir) if filename.endswith('.jpg')]
        self.tir_paths = [os.path.join(tir_dir, filename) for filename in os.listdir(tir_dir) if filename.endswith('.jpg')]

        if shuffle:
            combined = list(zip(self.rgb_paths, self.tir_paths))
            random.shuffle(combined)
            self.rgb_paths, self.tir_paths = zip(*combined)

        self.nSamples = len(self.rgb_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        rgb_path = self.rgb_paths[index]
        rgb_name = os.path.basename(rgb_path)
        gt_path = os.path.join(
            self.gt_dir, os.path.splitext(rgb_name)[0] + '.h5')
        tir_path = self.tir_paths[index]
        rgb_img, tir_img, target = load_data(rgb_path, tir_path, gt_path, self.train)

        if self.transform_rgb is not None and self.transform_tir is not None:
            rgb_img = self.transform_rgb(rgb_img)
            tir_img = self.transform_tir(tir_img)

        img = torch.cat((rgb_img, tir_img), dim=0)  # 将两个3通道图像合并成一个6通道图像

        return img, target


# 参数设置
lr = 1e-7
original_lr = lr
batch_size = 1  # 修改为4，加快训练
momentum = 0.95
decay = 5*1e-4
epochs = 400
steps = [-1, 1, 100, 150]
scales = [1, 1, 1, 1]   # TODO2:scales是否需要调整？
workers = 4
seed = time.time()
print_freq = 30
img_dir = "./dataset/train/rgb/"
# 新增红外图像路径 Thermal Imaging Radiometer
tir_dir = "./dataset/train/tir/"
gt_dir = "./dataset/train/hdf5s/"
pre = None
task = ""


def main():
    start_epoch = 0
    best_prec1 = 1e6

    torch.cuda.manual_seed(seed)

    model = CSRNet()

    model = model.cuda()

    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=decay)
    transform_rgb = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    # 这个transformer正确吗？
    transform_tir = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    # TODO1:这里只用到了RGB图像
    dataset = ImgDataset(
        img_dir, tir_dir,
        gt_dir,transform_rgb=transform_rgb, transform_tir=transform_tir,train=True
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    # pre 是什么的路径？ checkpoint!
    if pre:
        if os.path.isfile(pre):
            print("=> loading checkpoint '{}'".format(pre))
            checkpoint = torch.load(pre)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(pre))

    MAEs = {}
    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, epoch)

        train(model, criterion, optimizer, epoch, train_loader)
        prec1 = validate(model, val_loader)

        MAEs[epoch] = prec1 # 统计MAE

        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, task)
    
    # 绘制MAEs曲线
    # 确保在使用 numpy 之前将张量移动到 CPU
    epochs_list_cpu = torch.tensor(list(MAEs.keys())).cpu().numpy()
    maes_list_cpu = torch.tensor(MAEs.values()).cpu().numpy()

    plt.plot(epochs_list_cpu, maes_list_cpu)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE of each epoch')

    if not os.path.exists('pic'):
        os.makedirs('pic')
    
    plt.savefig('pic/MAEs.png')
    


def train(model, criterion, optimizer, epoch, train_loader):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' %
          (epoch, epoch * len(train_loader.dataset), lr))

    model.train()
    end = time.time()

    for i, (img, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        loss = criterion(output, target)

        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses))


def validate(model, val_loader):
    print('begin test')

    model.eval()
    mae = 0

    for i, (img, target) in enumerate(val_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        mae += abs(output.data.sum() -
                   target.sum().type(torch.FloatTensor).cuda())

    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '
          .format(mae=mae))

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
