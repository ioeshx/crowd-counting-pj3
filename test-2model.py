import torch
from model import CSRNet
from PIL import Image
from torchvision import transforms
from torch.autograd import Variable



rgb_paths = [f"./dataset/test/rgb/{i}.jpg" for i in range(1, 1001)]
tir_paths = [f"./dataset/test/tir/{i}R.jpg" for i in range(1,1001)]

model_rgb = CSRNet()
model_tir = CSRNet()
model_rgb.cuda().eval()
model_tir.cuda().eval()

checkpoint_rgb = torch.load('./model/rgb-model_best.pth.tar')
checkpoint_tir = torch.load('./model/tir-model_best.pth.tar')
checkpoint = torch.load('./model/best_weights.pth')

model_rgb.load_state_dict(checkpoint_rgb['state_dict'])
model_tir.load_state_dict(checkpoint_tir['state_dict'])
alpha = torch.tensor(checkpoint['alpha'], device='cuda')
beta = torch.tensor(checkpoint['beta'], device='cuda')


# 定义图像预处理
transform_rgb = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_tir = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


for i in range(len(rgb_paths)):
    img_rgb = transform_rgb((Image.open(rgb_paths[i]).convert('RGB')))
    img_rgb = img_rgb.cuda()
    img_rgb = Variable(img_rgb)
    
    img_tir = transform_tir((Image.open(tir_paths[i]).convert('RGB')))
    img_tir = img_tir.cuda()
    img_tir = Variable(img_tir)
    
    # 模型前向传播
    with torch.no_grad():
        output_rgb = model_rgb(img_rgb.unsqueeze(0))
        output_tir = model_tir(img_tir.unsqueeze(0))
    
    output = alpha * output_rgb + beta * output_tir
    
    # 计算总数并打印结果
    ans = output.detach().cpu().sum()
    ans = "{:.2f}".format(ans.item())
    print(f"{i+1},{ans}")