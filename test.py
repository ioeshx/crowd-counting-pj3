import PIL.Image as Image
import torchvision.transforms.functional as F
import torch
from model import CSRNet
from torchvision import transforms
from torch.autograd import Variable

test_path = "./dataset/test/rgb/"
test_tir_path = "./dataset/test/tir/"
img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]
tir_paths = [f'{test_tir_path}{i}R.jpg' for i in range(1, 1001)]

model = CSRNet()
model = model.cuda()
checkpoint = torch.load('./model/model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

# 旧版本的图像预处理和推断代码：
# for i in range(len(img_paths)):
#     img = 255.0 * F.to_tensor(Image.open(img_paths[i]).convert('RGB'))

#     img[0, :, :] = img[0, :, :]-92.8207477031
#     img[1, :, :] = img[1, :, :]-95.2757037428
#     img[2, :, :] = img[2, :, :]-104.877445883
#     img = img.cuda()
#     output = model(img.unsqueeze(0))
#     ans = output.detach().cpu().sum()
#     ans = "{:.2f}".format(ans.item())
#     print(f"{i+1},{ans}")

transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
        0.229, 0.224, 0.225]),
])
# 红外光的transformer
transform_tir = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

for i in range(len(img_paths)):
    rgb_img = transform_rgb(Image.open(img_paths[i]).convert('RGB'))
    tir_img = transform_tir(Image.open(tir_paths[i]).convert('RGB'))

    # Concatenate RGB and TIR images along the channel dimension
    img = torch.cat((rgb_img, tir_img), dim=0)

    img = img.cuda()
    img = Variable(img)
    output = model(img.unsqueeze(0))
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")
