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




def save_checkpoint(state, is_best, task_id, filename='checkpoint.pth.tar', save_dir='./model/'):  # 添加保存目录参数
    checkpoint_path = os.path.join(save_dir, task_id + filename)
    torch.save(state, checkpoint_path)
    if is_best:
        best_model_path = os.path.join(
            save_dir, task_id + 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_model_path)


def load_data(img_path, gt_path, train=True):
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    target = cv2.resize(
        target, (target.shape[1]//8, target.shape[0]//8), interpolation=cv2.INTER_CUBIC)*64
    return img, target


class ImgDataset(Dataset):
    def __init__(self, img_dir, gt_dir, shape=None, shuffle=True, transform=None, train=False, batch_size=1, num_workers=4):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.transform = transform  # 用于数据增强的transformer
        self.train = train
        self.shape = shape
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.img_paths = [os.path.join(img_dir, filename) for filename in os.listdir(
            img_dir) if filename.endswith('.jpg')]

        if shuffle:
            random.shuffle(self.img_paths)

        self.nSamples = len(self.img_paths)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        img_path = self.img_paths[index]
        img_name = os.path.basename(img_path)
        gt_path = os.path.join(
            self.gt_dir, os.path.splitext(img_name)[0] + '.h5')
        img, target = load_data(img_path, gt_path, self.train)
        if self.transform is not None:
            img = self.transform(img)
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
# 训练中断时，可以从中间继续
pre = None
task = ""


def main():
    start_epoch = 0
    best_prec1 = 1e6

    torch.cuda.manual_seed(seed)

    model = CSRNet()

    model = model.cuda()
    # TODO3: criterion, optimizer和transform需要优化吗？
    # 超参数优化？
    criterion = nn.MSELoss(size_average=False).cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=decay)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    # TODO1:这里只用到了RGB图像
    dataset = ImgDataset(
        img_dir,
        gt_dir, transform=transform, train=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
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
    maes_list_cpu = torch.tensor(list(MAEs.values())).cpu().numpy()

    plt.plot(epochs_list_cpu, maes_list_cpu)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('MAE of each epoch')

    if not os.path.exists('pic'):
        os.makedirs('pic')
    # 获取当前时间
    current_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    plt.savefig('pic/MAEs-{current_time_str}.png'.format(current_time_str=current_time_str))


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
