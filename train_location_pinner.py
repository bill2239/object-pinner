import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from dataset import PhoneDataset, Rescale, ToTensor
from torch.autograd import Variable
import numpy as np
import os
import argparse
from mymodel import AlexNet
__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}




def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)

def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet']))

    return model


def distance(outputs, labels):
    # calculate distance between predicted output and label (gt) 
    diff = outputs.data - labels.data
    sq_diff = torch.mul(diff,diff)
    sq_sum= torch.sum(sq_diff,1)

    dis= torch.sqrt(sq_sum).numpy()
    return dis

def get_labels_map(labels):
     # labels_map : heat map of phone location
    labels_loc= (27*labels).byte()
    
    labels_map_idx = np.zeros(labels_loc.size(0))
    for j in range(labels_loc.size(0)):
        
        labels_map_idx[j] = labels_loc[j][0]*labels_loc[j][1]
    
    labels_map = torch.from_numpy(labels_map_idx).long()
    return labels_map

def eval(dataloader,model_path):
    running_corrects = 0
    net= torch.load(model_path)
    for i, sample in enumerate(dataloader):
        inputs= sample['image'].float() 
        labels = sample['location'].float()
        inputs= Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        outputs = net(inputs)
        dis = distance(outputs.cpu(),labels.cpu())
        running_corrects += len(dis[dis < 0.05])
    accuracy = running_corrects / float(len(dataloader.dataset))
    print "accuracy of the dataset is ",accuracy
    
    return accuracy

def train_alexnet(trainloader,valloader,max_epoch,pretrained_file= None):
    if pretrained_file:
        net = torch.load(pretrained_file).cuda()
    else:
        net=alexnet(pretrained=False, num_classes=2).cuda()
        #net.apply(weights_init)
    criterion = nn.L1Loss()
    criterion_softmax = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    for epoch in range(max_epoch):  # loop over the dataset multiple times

        running_loss = 0.0
        running_corrects = 0
        
        for i, sample in enumerate(trainloader):
            #sample = traindataset[i]

            print(i, type(sample['image']), type(sample['location']))
            # get the inputs
            # print i 
            inputs= sample['image'].float() 
            labels = sample['location'].float() 
            
             # labels_map : heat map of phone location
            
            labels_map = get_labels_map(labels)
            
            
            
            # zero the parameter gradients
            optimizer.zero_grad()
            inputs= Variable(inputs).cuda()

            labels = Variable(labels).cuda()
            labels_map = Variable(labels_map).cuda()
            # forward + backward + optimize
            # outputs : total network forward
            outputs = net(inputs)
            # outputs_map : predicted heat map of phone location
            outputs_map= net.forward_map(inputs)

            #cross entropy loss for heat map prediction
            loss_softmax = criterion_softmax(outputs_map,labels_map)
            loss_softmax.backward(retain_graph=True) # retain_variables is replaced by retain_graph for 0.4.0
            
            loss = criterion(outputs, labels) + loss_softmax
            loss.backward()
            optimizer.step()
            #print loss
            # print statistics
            running_loss += loss.data
            dis = distance(outputs.cpu(),labels.cpu())
            #print i 
            running_corrects += len(dis[dis < 0.05])
            if i % 20 == 19:    # print every 20 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss.cpu().numpy() / 20))
                print 'train accuracy', i+1 , running_corrects /float(20*4)
                running_loss = 0.0
        if epoch % 10 == 9:
            model_name = 'model_'+str(289)+'_0.0001_'+str(epoch)+'.pth'
            torch.save(net, model_name)
            eval(valloader, model_name)



    print('Finished Training')




if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('path')
    args = parser.parse_args()
    
    traindataset= PhoneDataset( label_path='labels_train.txt',
                           root_dir=args.path,
                           transform=transforms.Compose([
                               Rescale((224,224)),
                               ToTensor()]))
    trainloader = torch.utils.data.DataLoader(traindataset,
                                             batch_size=4, shuffle=False,
                                             )

    
    
    valdataset= PhoneDataset( label_path='labels_val.txt',
                           root_dir=args.path,
                           transform=transforms.Compose([
                               Rescale((224,224)),
                               ToTensor()]))
    valloader = torch.utils.data.DataLoader(traindataset,
                                             batch_size=1, shuffle=False,
                                             )
    
    train_alexnet(trainloader,valloader,100,pretrained_file="model_289_0.0001_79.pth")
    #train_alexnet(trainloader)
    #eval(trainloader,"model_289_0.0001_79.pth") #'find_phone_task/find_phone/'

    
