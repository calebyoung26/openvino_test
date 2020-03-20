import numpy as np
from torchvision.models.resnet import resnet18
import torchvision.transforms as transforms
from PIL import Image
import torch
import time

class Pytorch_test():
    
    def __init__(self):
        pass

    def run(self, input_img,number_iter=1):
        imagenet_mean = (0.485, 0.456, 0.406)
        imagenet_std  = (0.229, 0.224, 0.225)

        transform_test_IMAGENET = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        transform_test = transform_test_IMAGENET

        net18 = resnet18(pretrained=True)
        
        net18.eval()
        
        im = Image.open(input_img)
        x = transform_test(im)
        x = x.unsqueeze(dim=0)

        t0 = time.time()
        for i in range(number_iter):
            res18 = net18(x)[0]
        infer_time = time.time()-t0
        #print(net18(x)[0].shape)
        values, indices = res18.max(0)
        values, indices = torch.topk(res18, 10)

        labels ="test_model.labels"
        if labels:
            with open(labels, 'r') as f:
                labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
        else:
            labels_map = None

        print('<<<<<<<<<<<RESULTS FOR PYTORCH>>>>>>>>>>>>')
        for i, probs in enumerate(res18):
            if (i<10):
                print( labels_map[indices[i]], values[i].item(), indices[i])
                print("{:.7f} label {}".format(values[i].item(), labels_map[indices[i]]))
        print('\n')
        print("PyTorch ran {} iterations in {} seconds".format(number_iter,infer_time))
        print('\n')