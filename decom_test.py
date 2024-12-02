import os
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        self.net1_conv0 = nn.Conv2d(4, channel, kernel_size * 3, padding=4, padding_mode='replicate')
        self.net1_convs = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            nn.Conv2d(channel, channel, kernel_size, padding=1, padding_mode='replicate'),
            nn.ReLU()
        )
        self.net1_recon = nn.Conv2d(channel, 4, kernel_size, padding=1, padding_mode='replicate')

    def forward(self, input_im):
        input_max = torch.max(input_im, dim=1, keepdim=True)[0]
        input_img = torch.cat((input_max, input_im), dim=1)
        feats0 = self.net1_conv0(input_img)
        featss = self.net1_convs(feats0)
        outs = self.net1_recon(featss)
        R = torch.sigmoid(outs[:, 0:3, :, :])
        L = torch.sigmoid(outs[:, 3:4, :, :])
        return R, L

def load_image(image_path):
    image = Image.open(image_path)
    image = np.array(image, dtype="float32") / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return torch.FloatTensor(image)

def save_image(tensor, path):
    image = tensor.squeeze().cpu().numpy()  # 移除多余的维度
    if len(image.shape) == 2:  # 如果是单通道图像
        # 对于单通道图像，直接保存
        image = np.clip(image * 255.0, 0, 255.0).astype('uint8')
        Image.fromarray(image, mode='L').save(path)  # 使用 'L' 模式保存灰度图像
    else:
        # 对于多通道图像，先转置再保存
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255.0, 0, 255.0).astype('uint8')
        Image.fromarray(image).save(path)

def main(image_path, model_path, output_dir,reflectance_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DecomNet().to(device)
    
    # 加载 .tar 格式的模型参数
    checkpoint = torch.load("checkpoints/Decom/9200.tar")
    model.load_state_dict(checkpoint)
    model.eval()
    if not os.path.exists(reflectance_dir):
        os.makedirs(reflectance_dir)

    image_files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
    for image_file in image_files:
        input_image_path = os.path.join(image_path, image_file)
        input_image = load_image(input_image_path).to(device)
        with torch.no_grad():
            R, L = model(input_image)

        save_image(R, os.path.join(reflectance_dir, f"reflectance_{image_file}"))
        save_image(L, os.path.join(output_dir, f"illumination_{image_file}"))



if __name__ == "__main__":
    image_path = "./use_decom_dataset/test"
    model_path = "checkpoints/Decom/9200.tar"
    output_dir = "./test_dataset/testA"
    reflectance_dir = "./use_decom_dataset/cache"
    main(image_path, model_path, output_dir, reflectance_dir)