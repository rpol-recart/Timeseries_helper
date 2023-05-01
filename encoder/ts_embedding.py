import pandas as pd
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
from tqdm import tqdm
from autoencoder_model.model import Autoencoder
from autoencoder_model.dataset import TimeSeriesDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def load_data(datadir):
    with open(os.path.join(datadir, 'dataset3.pkl'), 'rb') as f:
        dd = pickle.load(f)
    
    return dd


def tokenize_ts(data, intervals):
    big_ls = []
    df = data[:-30].copy()
    for sym in ['OBJ1T', 'OBJ2T', 'OBJ3T', 'TEMP']:
        df[sym+'_delta'] = df[sym+'_delta'].bfill()
        df[sym+'_int'] = pd.cut(df[sym], bins=intervals)
        start_int = df[sym+'_int'].iloc[0]
        ls1 = []

        pos = 0
        for ind, i in df.iterrows():
            if i[sym+'_int'] == start_int:
                ls1.append(i[sym+'_delta'])
            else:
                big_ls.append(
                    [key, sym, pos, i[sym+'_int'].right, np.array(ls1)])
                pos += 1
                start_int = i[sym+'_int']
                ls1 = []
                ls1.append(i[sym+'_delta'])
    return big_ls


def make_dataset(data_lst):
    lst = []
    for data in data_lst:
        lst.append(data[4])
    dataset = TimeSeriesDataset(np.array(lst))
    return dataset


if __name__ == '__main__':
    # params
    DATA_DIR = 'dataset_ts/'
    CALCULATE_EMB = False
    input_dim = 1
    hidden_size = 32
    latent_size = 5

    if CALCULATE_EMB:
        dd = load_data(DATA_DIR)
        intervals = pd.interval_range(300, 1100, freq=2)
        long_list = []

        for key in tqdm(dd.keys(), total=len(dd.keys())):
            token_lst = tokenize_ts(dd[key], intervals)
            long_list = long_list+token_lst
        # save data
        with open('dataset_ts/emb_dataset.pkl', 'wb') as f:
            pickle.dump(long_list, f)
    else:
        with open('dataset_ts/emb_dataset.pkl', 'rb') as f:
            long_list = pickle.load(f)

    model = Autoencoder(input_dim, hidden_size, latent_size)

    dataset = make_dataset(long_list)
    # задаем размеры train и test датасетов в процентах от исходного
    train_size = 0.8
    test_size = 0.2

    # вычисляем размеры train и test датасетов
    num_data = len(dataset)
    train_num = int(train_size * num_data)
    test_num = num_data - train_num

    # разбиваем датасет на train и test
    train_dataset, test_dataset = random_split(dataset, [train_num, test_num])
    # train_dataset = AugmentedDataset(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1,
                             shuffle=False, num_workers=4)

    # train_process
    import torch.optim as optim

    # Определение функции потерь и оптимизатора
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    loss_list = []
    best_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Тренировочный цикл
    num_epochs = 10

    for epoch in range(num_epochs):
        # Обнуляем градиенты
        running_loss = 0.0
        model.train()
        for data in tqdm(train_dataloader):

            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Обнуляем градиенты
            optimizer.zero_grad()

            # Прогоняем входные данные через автоэнкодер
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # Обратный проход и оптимизация параметров
            loss.backward()
            optimizer.step()

            # Суммируем потери по батчам
            running_loss += loss.item() * inputs.size(0)
        # test loop
        val_loss = 0
        model.eval()
        with torch.no_grad():
            test_loss = 0
            for data in test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
            test_loss /= len(test_dataset)
            print('Test Loss: {:.6f}'.format(test_loss))
        # Выводим средние потери за эпоху
        epoch_loss = running_loss / len(train_dataset)
        print('Epoch [{}/{}], Loss: {:.6f}'.format(epoch +
                                                   1, num_epochs, epoch_loss))
        if epoch > 1:
            if test_loss < best_loss:
                best_loss = test_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }
                torch.save(checkpoint, 'checkpointEnc2_' +
                           str(epoch).zfill(3)+'.pth')

        loss_list.append([epoch, running_loss, test_loss])
    with open('autoencoder_model/loss_lst.pkl', 'wb') as f:
        pickle.dump(loss_list, f)
