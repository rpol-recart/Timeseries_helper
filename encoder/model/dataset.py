import torch
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Датасет временных рядов"""

    def __init__(self, data, train=False):
        """
        Аргументы:
        data -- pandas.DataFrame, датафрейм со временными рядами
        train -- bool, флаг, указывающий, является ли датасет тренировочным
        """
        self.data = data

        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Извлекаем временной ряд по индексу
        time_series = self.data[idx].astype('float32')
        label = self.data[idx].astype('float32')

        # Возвращаем аугментированный временной ряд и оригинальный для вычисления потерь
        return torch.tensor(time_series).unsqueeze(1), torch.tensor(label).unsqueeze(1)
