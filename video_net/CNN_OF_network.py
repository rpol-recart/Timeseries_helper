import torch
import torch.nn as nn


class MultiChannelCNN(nn.Module):
    def __init__(self, num_frames, height, width, num_classes):
        super(MultiChannelCNN, self).__init__()

        # Определяем сверточные слои для видео
        self.video_cnn = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Определяем сверточные слои для оптического потока
        self.optical_flow_cnn = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten()
        )

        # Полносвязные слои для классификации
        self.fc_layers = nn.Sequential(
            nn.Linear(2 * 32 * (height // 4) * (width // 4), 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, video, optical_flow):
        # Пропускаем видео через сверточные слои
        video_features = self.video_cnn(video)

        # Пропускаем оптический поток через сверточные слои
        optical_flow_features = self.optical_flow_cnn(optical_flow)

        # Объединяем признаки по последней оси
        merged_features = torch.cat(
            (video_features, optical_flow_features), dim=1)

        # Пропускаем объединенные признаки через полносвязные слои для классификации
        output = self.fc_layers(merged_features)

        return output
