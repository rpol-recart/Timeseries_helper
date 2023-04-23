import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, num_frames, height, width, channels, num_classes):
        super(Net, self).__init__()

        # Определяем сверточные слои для видео
        self.conv_video = nn.Conv2d(channels, 32, kernel_size=3, padding=1)
        self.pool_video = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_video = nn.Flatten()

        # Определяем сверточные слои для оптического потока
        self.conv_optical_flow = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.pool_optical_flow = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten_optical_flow = nn.Flatten()

        # Определяем полносвязные слои для классификации
        self.dense1 = nn.Linear(32*int(height/4)*int(width/4) * 2, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, input_video, input_optical_flow):
        # Извлекаем признаки из видео
        x_video = F.relu(self.conv_video(input_video))
        x_video = self.pool_video(x_video)
        x_video = self.flatten_video(x_video)

        # Извлекаем признаки из оптического потока
        x_optical_flow = F.relu(self.conv_optical_flow(input_optical_flow))
        x_optical_flow = self.pool_optical_flow(x_optical_flow)
        x_optical_flow = self.flatten_optical_flow(x_optical_flow)

        # Объединяем признаки из видео и оптического потока
        merged_features = torch.cat([x_video, x_optical_flow], dim=1)

        # Классификация событий
        x = F.relu(self.dense1(merged_features))
        x = self.output(x)
        output = F.softmax(x, dim=1)

        return output


# Определяем размеры входных данных и число классов
num_frames = 10
height = 256
width = 256
channels = 3
num_classes = 2

# Создаем экземпляр модели
model = Net(num_frames, height, width, channels, num_classes)

# Определяем функцию потерь и оптимизатор
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Генерируем случайные данные
input_video = torch.randn(2, num_frames, channels, height, width)
input_optical_flow = torch.randn(2, num_frames, 2, height, width)

# Выполняем прямой проход через модель
output = model(input_video, input_optical_flow)

# Выводим размерность выхода и число параметров модели
print(output.size())
print(sum(p.numel() for p in model.parameters()))
