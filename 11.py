import torch
import torch.nn as nn

class FullModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积模块
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1),  # 输入1通道，输出8通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 58x58 → 29x29
            nn.Conv2d(8, 32, kernel_size=5, stride=2),  # 29x29 → 13x13
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13x13 → 6x6
        )

        # 后续处理层
        self.fc1 = nn.Linear(36, 36)  # 输入36（6*6），输出32
        self.ln1 = nn.LayerNorm(36)  # 对最后一个维度（32）归一化
        self.fc2 = nn.Linear(32, 32)  # 输入32，输出36
        self.ln2 = nn.LayerNorm(32)
        self.relu = nn.ReLU()

    def forward(self, x,batch_size):
        # 1. 卷积部分
        x = self.conv_block(x)  # [ 32, 6, 6]

        # 2. Reshape: [ batch_size,32, 6*6] → [ 32, 36]
        x = x.reshape(batch_size,32,36)

        # 3. 第一个全连接层（输入32，输出32）
        x = self.fc1(x.view(-1, 36))
        x = x.view(batch_size, 32, 36)  # 重新调整为 [batch_size, 32, 36]
        x = self.ln1(x)
        x = self.relu(x)

        # 4. 转置 → [batch_size,36, 32]
        x = x.transpose(1, 2)  # 交换维度1和2

        # 5. 第二个全连接层（输入32，输出36）
        x = self.fc2(x.reshape(-1,32))  # [batch, 36, 32]
        x = x.view(batch_size, 36, 32)  # 重新调整为 [batch_size, 36, 32]
        x = self.ln2(x)
        x = self.relu(x)

        return x


def test_model():
    # 创建模型实例
    model = FullModel()

    # 假设输入是一个 batch_size=8 的 60x60 的单通道图像
    input_tensor = torch.randn(8, 1, 60, 60)  # batch_size=8, 单通道图像 60x60

    # 将输入传入模型
    output = model(input_tensor,batch_size=8)

    # 输出结果的形状
    print(f'Output shape: {output.shape}')


# 运行测试
test_model()