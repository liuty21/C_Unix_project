from load_mnist import Dataload
from model_AlexNet import Model_Alexnet
from model_ResNet import ResNet
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt

train_img_file = '../MNIST/train-images.idx3-ubyte'
train_label_file = '../MNIST/train-labels.idx1-ubyte'
test_img_file = '../MNIST/t10k-images.idx3-ubyte'
test_label_file = '../MNIST/t10k-labels.idx1-ubyte'

batch_size = 100
lr = 0.001
epochs = 10
show_results = False
do_test = False


def main():
    # model = Model_Alexnet()
    model = ResNet()
    model.cuda()

    data_train = Dataload(train_img_file, train_label_file)
    data_test = Dataload(test_img_file, test_label_file, split='test')
    train_loader = torch.utils.data.DataLoader(
        data_train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        data_test, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr, 
                                betas=(0.9,0.999), eps=1e-8,
                                weight_decay=1e-4)

    if do_test:
        state = torch.load('./Model_Alexnet.pth')
        model.load_state_dict(state)
        test(test_loader, model, show_results=show_results)
        return

    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        test(test_loader, model, show_results=show_results)

    # torch.save(model.state_dict(), './Model_Alexnet.pth')
    torch.save(model.state_dict(), './Model_resnet.pth')

    # for params in model.state_dict():
    #     print(params, model.state_dict()[params].size())


def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    for i, (img, label) in enumerate(train_loader):
        img = img.cuda()
        label = label.cuda()

        img = torch.autograd.Variable(img, requires_grad = True)
        label = torch.autograd.Variable(label, requires_grad = False)

        output = model(img)

        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i%100 == 99:
            print('Epoch:%d, step:%d, loss=%.4f'%(epoch+1,i+1, loss))



def test(test_loader, model, show_results=False):
    correct = 0
    model.eval()
    if show_results:
        plt.figure()

    for i, (img, label) in enumerate(test_loader):
        img = img.cuda()
        label = label.cuda()
        
        img = torch.autograd.Variable(img, requires_grad = False)
        label = torch.autograd.Variable(label, requires_grad = False)

        with torch.no_grad():
            output = model(img)
        # print(img[0].size())

        pred = output.argmax(dim=1, keepdim=True)

        correct += pred.eq(label.view_as(pred)).sum().item()

        if show_results:
            print('Predict:',pred[0].item(), 'Label:',label[0].item())
            plt.imshow(img[0].numpy().reshape((28,28)),'gray')
            plt.pause(0.1)
            plt.show()

    accuracy = correct/len(test_loader.dataset)

    print('Test accuracy = {}%'.format(accuracy*100))

    return accuracy

if __name__ == '__main__':
    main()