import torch
import torch.nn as nn
import os
import torchvision.datasets as datasets
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
"""
Methods for dealing with imbalanced datasets:
1. Oversampling  - 적은 클래스의 데이터를 늘리는 방법
2. Class weighting - 클래스의 편향된 분포를 고려하여 현재 훈련 알고리즘의 loss function의 penalty 수정. 즉, 적은 클래스에 더 높은 가중치를 주는것
"""

# loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1, 50]))
# 두 개의 클래스의 비중 골든리트 리버 50, 스웨디시 1이기 때문에 스웨디시에 50배를 곱한 값을 갖도록 함

def get_loader(root_dir, batch_size):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ]
    )

    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)

    # class_weights = [1,50] # ==[1/50,1]
    class_weights = []
    for root, subdir, files in os.walk(root_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))
    sample_weights = [0]* len(dataset) # initial

    for idx, (data, label) in enumerate(dataset):
        class_weight = class_weights[label] # 0번째 =골든리트리버 = Class weight 1, 1번째 = 스웨디시 = Class weight 50
        sample_weights[idx] = class_weight
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(dataset), replacement=True)#false -> 한 번만 실행됨 결과를 얻을 수 없음
    loader =DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return loader

def main():
    loader = get_loader(root_dir='dataset/Imbalanced_dataset', batch_size=8)

    num_retrivers=0
    num_elkhounds=0
    for epoch in range(10):
        for data, labels in loader:
            num_retrivers += torch.sum(labels==0)
            num_elkhounds += torch.sum(labels == 1)

    print(num_retrivers, num_elkhounds)
if __name__ == '__main__':
    main()
