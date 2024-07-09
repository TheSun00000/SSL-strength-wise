import os
import torchvision
    
train_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10', train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10('./dataset/cifar10', train=False, download=True)

# train_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=True, download=True)
# test_dataset = torchvision.datasets.CIFAR100('./dataset/cifar100', train=False, download=True)

# train_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='train', download=True)
# test_dataset = torchvision.datasets.SVHN('./dataset/SVHN/', split='test', download=True)

# train_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='train')
# train_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='unlabeled')
# test_dataset = torchvision.datasets.STL10('./dataset/STL10/', split='test')



# if 'tiny-imagenet-200' in os.listdir('./dataset/'):
#     # This method is responsible for separating validation images into separate sub folders

#     dataset_dir = "./dataset/tiny-imagenet-200"
#     val_dir = os.path.join(dataset_dir, 'val')
#     img_dir = os.path.join(val_dir, 'images')

#     fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
#     data = fp.readlines()
#     val_img_dict = {}
#     for line in data:
#         words = line.split('\t')
#         val_img_dict[words[0]] = words[1]
#     fp.close()

#     # Create folder if not present and move images into proper folders
#     for img, folder in val_img_dict.items():
#         newpath = (os.path.join(img_dir, folder))
#         if not os.path.exists(newpath):
#             os.makedirs(newpath)
#         if os.path.exists(os.path.join(img_dir, img)):
#             os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))