import numpy as np
import struct
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

matplotlib.use('TkAgg')

train_img_file = '../MNIST/train-images.idx3-ubyte'
train_label_file = '../MNIST/train-labels.idx1-ubyte'

# decode image file
def decode_idx3(idx3_file):
    bin_data = open(idx3_file, 'rb').read()

    offset = 0
    fmt_header = '>iiii'
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('magic_number:%d, num_images: %d, image_size: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print(offset)
    fmt_image = '>' + str(image_size) + 'B'
    # print(fmt_image,offset,struct.calcsize(fmt_image))
    images = np.empty((num_images, num_rows, num_cols))
    # plt.figure()

    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('Unpack %d' % (i + 1) + 'images')
            # print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)
        # plt.imshow(images[i],'gray')
        # plt.pause(0.00001)
        # plt.show()
    # print(images[0])

    return images, num_images

# decode label file
def decode_idx1(idx3_file):

    bin_data = open(idx3_file, 'rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('magic_number:%d, num_images: %d' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('Unpack %d' % (i + 1) + 'labels')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    # print(labels[0])
    return labels, num_images


class Dataload(data.Dataset):
    """load MNIST dataset"""
    def __init__(self, imgpath, labelpath, split = 'train'):
        self.images, self.imgnum = decode_idx3(imgpath)
        self.labels, labelnum = decode_idx1(labelpath)
        self.transform = transforms.ToTensor()
        assert self.imgnum==labelnum
        print("load MNIST %s Dataset with %d images"%(split, self.imgnum))
        
    def __getitem__(self, index):
        img, label = self.images[index], int(self.labels[index])
        img = self.transform(img).float()
        return img, label

    def __len__(self):
        return self.imgnum

if __name__ == '__main__':
    decode_idx3(train_img_file)
    # train_dataset = Dataload(train_img_file, train_label_file)