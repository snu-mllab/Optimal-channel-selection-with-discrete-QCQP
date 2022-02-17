import torchvision.transforms as transforms

def cifar10_transform():
    mean = [x / 255 for x in [125.3, 123.0, 113.9]]
    std = [x / 255 for x in [63.0, 62.1, 66.7]]

    # Data Load
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)]) 

    return transform_train, transform_test


