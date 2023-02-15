import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import random
import os
from TinyImageNet import TinyImageNet



class OpenDatasets:
    def __init__(self, train_batch_size, test_batch_size=0, num_workers=0, add_noisy_instances=False, data_dir='./data/'):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size if test_batch_size > 0 else train_batch_size
        self.num_workers=0
        self.data_dir=data_dir

        self.lable_map=None

        self.add_noisy_instances=add_noisy_instances

        self.datasetsnames = ["svhn", "mnist", "emnist", "cifar10", "cifar100", "tinyimagenet"]
        self.dataset_dict = {
            "svhn_train" : self.get_svhn_train,
            "svhn_test"  : self.get_svhn_test,
            "mnist_train": self.get_mnist_train,
            "mnist_test" : self.get_mnist_test,
            "emnist_train": self.get_emnist_train,
            "emnist_test" : self.get_emnist_test,
            "cifar10_train": self.get_cifar10_train,
            "cifar10_test" : self.get_cifar10_test,
            "cifar100_train": self.get_cifar100_train,
            "cifar100_test" : self.get_cifar100_test,
            "tinyimagenet_train": self.get_tinyimagenet_train,
            "tinyimagenet_test" : self.get_tinyimagenet_test,
        }

    def combine_datasets(self, dataloader1, dataloader2):
        dataset1 = dataloader1.dataset
        dataset2 = dataloader2.dataset
        return self.train_DataLoader(torch.utils.data.ConcatDataset([dataset1, dataset2]))

    def get_dataset_byname(self, name, train=True, specific_classes=None):
        if name not in self.datasetsnames:
            raise ValueError(f"{name} is an invalid dataset. Try {self.datasetsnames}")
        dataset_name = name + "_" + ("train" if train else "test")



        return self.dataset_dict[dataset_name](specific_classes)
       
    def get_dataset_class_count(self, name):
        if name not in self.datasetsnames:
            raise f"{name} is an invalid dataset. Try {self.datasetsnames}"
        ret_dict ={
            "svhn": 10,
            "mnist": 10,
            "emnist": 15,
            "cifar10": 10,
            "cifar100": 100,
            "tinyimagenet": 200,
        }
        return ret_dict[name]

    def set_random_classes(self,seed, inlier_count=6, total_classes=0, classes=None):
        random.seed(seed)
        if classes == None: classes = list(range(0,total_classes))
        random.shuffle(classes)

        labl_map = [-1] * len(classes)
        for i, c in enumerate(classes):
            labl_map[c] = i
        self.lable_map = labl_map
        self.inliers = classes[:inlier_count]
        self.outliers = classes[inlier_count:]

        return self.inliers, self.outliers

    def filter_specific_classes(self, class_list, input_data, labels, lmap = None):
        if lmap is None: lmap = self.lable_map
        if lmap is None: raise "cannot filter classes unless set_random_classes is called"

        ret_data = []
        ret_labels = []

        for data, label in zip(input_data, labels):
            if label in class_list:
                ret_data.append(data)
                ret_labels.append(lmap[label])

        return ret_data, ret_labels

    def filter_wrapper(specific_classes, data, labels):
        print("Warning, this function has not been tested")
        if specific_classes is not None:
            new_data, new_labels = filter_specific_classes(specific_classes, dataset.data, dataset.labels)

            data   = new_data
            labels = new_labels

    def get_svhn_transform():
        return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def get_cifar_transform(train=False):

        r_mean = 125.3/255
        g_mean = 123.0/255
        b_mean = 113.9/255
        r_std = 63.0/255
        g_std = 62.1/255
        b_std = 66.7/255

        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding = 4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
        ])

        test_transform = transforms.Compose([
            transforms.CenterCrop((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((r_mean, g_mean, b_mean), (r_std, g_std, b_std)),
        ])

        return train_transform if train else test_transform

       


    def get_mnist_transform():
        return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize((0.1307,), (0.3081,))
        ])

    def _DataLoader(self, dataset, batch_size, do_shuffle=False):


        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers = self.num_workers, shuffle=do_shuffle, pin_memory = True)


    def train_DataLoader(self, dataset):


        return self._DataLoader(dataset, self.train_batch_size, do_shuffle=True)

    def test_DataLoader(self, dataset):
        return self._DataLoader(dataset, self.test_batch_size, do_shuffle=False)

    def get_svhn_train(self, specific_classes=None):
        dataset = datasets.SVHN(self.data_dir+'svhn', split='train', download=True,
                       transform=OpenDatasets.get_svhn_transform())

        label_count = 10

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.labels)

            dataset.data   = new_data
            dataset.labels = new_labels
            label_count = len(specific_classes)



        if self.add_noisy_instances:
            instance_count = int(len(dataset.data))
            noisy_shape = dataset.data[0].shape
            datatoadd = []
            for _ in range(instance_count):
                datatoadd.append(np.random.randint(0, 255, noisy_shape, dtype=np.uint8))

            dataset.data += datatoadd
            dataset.labels += [label_count] * instance_count


        return self.train_DataLoader(dataset)


    def get_svhn_test(self, specific_classes=None):
        dataset = datasets.SVHN(self.data_dir+'svhn', split='test',download=True, transform=OpenDatasets.get_svhn_transform())
        
        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.labels)

            dataset.data   = new_data
            dataset.labels = new_labels


        return self.test_DataLoader(dataset)





    def get_mnist_train(self, specific_classes=None):
        dataset = datasets.MNIST(self.data_dir+'mnist', train=True, download=True,
                       transform=OpenDatasets.get_mnist_transform())

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels

        if self.add_noisy_instances: exit("cant add noisy to this dataset yet")

        return self.train_DataLoader(dataset)

    def get_mnist_test(self, specific_classes=None):
        dataset    = datasets.MNIST(self.data_dir+'mnist', train=False, download=True, transform=OpenDatasets.get_mnist_transform())

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels

        return self.test_DataLoader(dataset)


    def get_emnist_train(self, specific_classes=None):
        dataset = datasets.EMNIST(self.data_dir+'mnist', train=True, download=True, split='letters',
                       transform=OpenDatasets.get_mnist_transform())

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels

        if self.add_noisy_instances: exit("cant add noisy to this dataset yet")

        return self.train_DataLoader(dataset)

    def get_emnist_test(self, specific_classes=None):
        dataset    = datasets.EMNIST(self.data_dir+'mnist', train=False, download=True, split='letters', transform=OpenDatasets.get_mnist_transform())

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels

        return self.test_DataLoader(dataset)





    def get_cifar10_train(self, specific_classes=None):
        dataset = datasets.CIFAR10(self.data_dir+'cifar', train=True, download=True,
                       transform=OpenDatasets.get_cifar_transform(train=True))


        label_count = len(dataset.classes)
        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels
            label_count = len(specific_classes)

        if self.add_noisy_instances:
            instance_count = int(len(dataset.data))
            noisy_shape = dataset.data[0].shape
            datatoadd = []
            for _ in range(instance_count):
                datatoadd.append(np.random.randint(0, 255, noisy_shape, dtype=np.uint8))

            dataset.data += datatoadd
            dataset.targets += [label_count] * instance_count


        return self.train_DataLoader(dataset)

    def get_cifar10_test(self, specific_classes=None):
        dataset    = datasets.CIFAR10(self.data_dir+'cifar', train=False, download=True, transform=OpenDatasets.get_cifar_transform())

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data    = new_data
            dataset.targets = new_labels

        return self.test_DataLoader(dataset)




    def get_cifar100_train(self, specific_classes=None):
        dataset = datasets.CIFAR100(os.path.join(self.data_dir, 'cifar100'), train=True, download=True,
                       transform=OpenDatasets.get_cifar_transform(train=True))

        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets)

            dataset.data   = new_data
            dataset.targets = new_labels

        if self.add_noisy_instances: exit("cant add noisy to this dataset yet")


        return self.train_DataLoader(dataset)

    def get_cifar100_test(self, specific_classes=None):
        dataset    = datasets.CIFAR100(os.path.join(self.data_dir, 'cifar100'), train=False, download=True, transform=OpenDatasets.get_cifar_transform())
        if specific_classes is not None:
            new_data, new_labels = self.filter_specific_classes(specific_classes, dataset.data, dataset.targets, lmap=self.cifar100map)

            dataset.data   = new_data
            dataset.targets = new_labels

        return self.test_DataLoader(dataset)

    cifar10_classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    cifar100_coarse_classes = ['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical devices','household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes','large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals','trees','vehicles 1','vehicles 2']
    cifar100_finer_classes  = [
        'beaver', 'dolphin', 'otter', 'seal', 'whale',
        'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
        'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
        'bottles', 'bowls', 'cans', 'cups', 'plates',
        'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
        'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
        'bed', 'chair', 'couch', 'table', 'wardrobe',
        'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
        'bear', 'leopard', 'lion', 'tiger', 'wolf',
        'bridge', 'castle', 'house', 'road', 'skyscraper',
        'cloud', 'forest', 'mountain', 'plain', 'sea',
        'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
        'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
        'crab', 'lobster', 'snail', 'spider', 'worm',
        'baby', 'boy', 'girl', 'man', 'woman',
        'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
        'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
        'maple', 'oak', 'palm', 'pine', 'willow',
        'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
        'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'
    ]

    animal_coarse_indices   = [0,1,7,8,11,12,13,14,15,16]
    fine_animal_indices     = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84]
    
    def _calc_fine_animal_indices():
        fine_animal_indices = []
        for i in animal_coarse_indices:
            c = list(range(i*5,i*5 + 5))
            fine_animal_indices.extend(c)
        for i in fine_animal_indices:
            print(cifar100_finer_classes[i])


    def get_random_cifar100(self, seed, num_open_classes):

        random.seed(seed)
        classes = list(range(len(OpenDatasets.cifar100_finer_classes)))
        random.shuffle(classes)

        self.cifar100map = [-1] * len(OpenDatasets.cifar100_finer_classes)
        for i, c in enumerate(classes): self.cifar100map[c] = i

        outliers = []
        i = 0
        while len(outliers) < num_open_classes:
            if classes[i] in OpenDatasets.fine_animal_indices:
                outliers.append(classes[i])
            i+=1

        cifar10_inliers = [0,1,8,9,2,3,4,5,6,7]
        labl_map = [-1] * len(cifar10_inliers)
        for i, c in enumerate(cifar10_inliers):
            labl_map[c] = i
        self.lable_map = labl_map

        print(f"Inliers of NOT Animals: {[OpenDatasets.cifar10_classes[i] for i in cifar10_inliers[:4]]}")
        print(f"Outliers of Animals: {[OpenDatasets.cifar100_finer_classes[i] for i in outliers]}")
        return  cifar10_inliers[:4], outliers

    

    def get_tinyimagenet_train(self, specific_classes=None):
        dataset = TinyImageNet(os.path.join(self.data_dir,"tiny-imagenet-200"), split='train', transform=OpenDatasets.get_cifar_transform(train=True), in_memory=True, specific_classes=specific_classes)


        if self.add_noisy_instances: exit("cant add noisy to this dataset yet")


        return self.train_DataLoader(dataset)

    def get_tinyimagenet_test(self, specific_classes=None):
        dataset = TinyImageNet(os.path.join(self.data_dir,"tiny-imagenet-200"), split='val', transform=OpenDatasets.get_cifar_transform(), in_memory=True, specific_classes=specific_classes)

        '''if specific_classes is not None:
            specific_classes.sort()
            all_needed_indices = []
            for c in specific_classes:
                cur_range = list(range(c*50, (c+1)*50))
                all_needed_indices.extend(cur_range)


            dataset = MySubset(dataset, all_needed_indices, specific_classes)'''

        return self.test_DataLoader(dataset)




class MySubset(torch.utils.data.Dataset):
        r"""
        Subset of a dataset at specified indices.

        Arguments:
            dataset (Dataset): The whole Dataset
            indices (sequence): Indices in the whole set selected for subset
        """
        def __init__(self, dataset, indices, specific_classes):
            self.dataset = dataset
            self.indices = indices
            self.label_mapper = {}
            for i in range(len(specific_classes)):
                self.label_mapper[specific_classes[i]] = i
            self.specific_classes = specific_classes

        def __getitem__(self, idx):
            if idx > len(self): 
                import pdb; pdb.set_trace()
            image, label = self.dataset[self.indices[idx]]
            if label not in self.specific_classes:
                import pdb; pdb.set_trace()
            return image, self.label_mapper[label]

        def __len__(self):
            return len(self.indices)