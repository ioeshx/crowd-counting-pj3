import cv2
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image
import matplotlib.pyplot as plt

# 打开图像文件
image = Image.open('./dataset/test/tir/500R.jpg')

# 确保图像是RGB模式
if image.mode != 'RGB':
    image = image.convert('RGB')

# 分离图像的三个通道
red_channel, green_channel, blue_channel = image.split()

# 获取每个通道的像素值
red_pixels = red_channel.getdata()
green_pixels = green_channel.getdata()
blue_pixels = blue_channel.getdata()

# 绘制直方图
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

axes[0].hist(list(red_pixels), bins=256, color='red', alpha=0.6)
axes[0].set_title('Red Channel')
axes[0].set_xlim(0, 255)
axes[0].set_xlabel('Pixel Value')
axes[0].set_ylabel('Frequency')

axes[1].hist(list(green_pixels), bins=256, color='green', alpha=0.6)
axes[1].set_title('Green Channel')
axes[1].set_xlim(0, 255)
axes[1].set_xlabel('Pixel Value')

axes[2].hist(list(blue_pixels), bins=256, color='blue', alpha=0.6)
axes[2].set_title('Blue Channel')
axes[2].set_xlim(0, 255)
axes[2].set_xlabel('Pixel Value')

plt.tight_layout()
plt.savefig("./pic/bin.png")
plt.show()




# image = Image.open('./dataset/test/tir/1R.jpg')
# mode = image.mode
# # 打印图像模式
# print(f"The mode of the image is: {mode}")

# if image.mode != 'RGB':
#     image = image.convert('RGB')

# # 分离图像的三个通道
# red_channel, green_channel, blue_channel = image.split()
# # 显示每个通道
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# axes[0].imshow(red_channel, cmap='Reds')
# axes[0].set_title('Red Channel')
# axes[0].axis('off')

# axes[1].imshow(green_channel, cmap='Greens')
# axes[1].set_title('Green Channel')
# axes[1].axis('off')

# axes[2].imshow(blue_channel, cmap='Blues')
# axes[2].set_title('Blue Channel')
# axes[2].axis('off')

# plt.savefig("./pic/1.png")





# # 判断图片是不是单通道
# img = Image.open("./dataset/test/tir/1R.jpg")
# print(len(img.split()))

# img = Image.open("./dataset/train/tir/1R.jpg")
# print(len(img.split()))

# # 虽然是黑白的，但是3通道的