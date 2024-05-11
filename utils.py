import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import torchvision.models as models
import torch.nn.functional as F
from PIL import Image
from collections import OrderedDict


def transform_img(data_dir):
  train_dir = data_dir + '/train'
  valid_dir = data_dir + '/valid'
  test_dir = data_dir + '/test'

  data_transforms = {
    'train_transforms': transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
    'test_transforms': transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]),
    'validation_transforms': transforms.Compose([transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                                  ])
  }

  image_datasets = {
    'train_data': datasets.ImageFolder(train_dir, transform=data_transforms['train_transforms']),
    'test_data': datasets.ImageFolder(test_dir, transform=data_transforms['test_transforms']),
    'valid_data': datasets.ImageFolder(valid_dir, transform=data_transforms['validation_transforms'])
  }

  return image_datasets['train_data'], image_datasets['valid_data'], image_datasets['test_data']


def load_data(data_dir):

  train_data, valid_data, test_data = transform_img(data_dir)

  dataloaders = {
    'trainloader': torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True),
    'testloader': torch.utils.data.DataLoader(test_data, batch_size = 32, shuffle = True),
    'validloader': torch.utils.data.DataLoader(valid_data, batch_size = 32, shuffle = True)
  }

  return dataloaders['trainloader'], dataloaders['validloader'], dataloaders['testloader']

def network_struct(arch, dropout, hidden_units, learning_rate, use_gpu):
  
  archs = {'vgg16': 25088, 'densenet121': 1024, 'alexnet': 9216}
  
  if arch == 'vgg16':
    model = models.vgg16(pretrained = True)        
  elif arch == 'densenet121':
    model = models.densenet121(pretrained = True)
  elif arch == 'alexnet':
    model = models.alexnet(pretrained = True)
  else:
    print(f'{arch} is not a valid model. Try one of these: vgg16, densenet121 or alexnet.')
      
  for param in model.parameters():
    param.requires_grad = False
      
  classifier = nn.Sequential(OrderedDict([
              ('fc1', nn.Linear(archs[arch], hidden_units)),
              ('relu1', nn.ReLU()),
              ('drop_out1',nn.Dropout(dropout)),
              ('fc2', nn.Linear(hidden_units, 128)),
              ('relu2', nn.ReLU()),
              ('drop_out2',nn.Dropout(dropout)),
              ('fc3', nn.Linear(128, 102)),
              ('output', nn.LogSoftmax(dim=1))
            ]))

  model.classifier = classifier
  criterion = nn.NLLLoss()
  optimizer = optim.Adam(model.classifier.parameters(), learning_rate)

  if use_gpu:
    model.cuda()
  
  return model, optimizer, criterion

def train_neural_net(data_dir, model, criterion, optimizer, trainloader, epochs, print_every, use_gpu):
  
  _, vd_dl, _ = load_data(data_dir)

  steps = 0
  
  if use_gpu:
    model.to('cuda')

  for e in range(epochs):
    running_loss = 0

    for ii, (images, labels) in enumerate(trainloader):
      steps += 1

      if use_gpu:
        images, labels = images.to('cuda'), labels.to('cuda')

      optimizer.zero_grad()

      # Forward and backward passes
      outputs = model.forward(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()

      if steps % print_every == 0:
        model.eval()
        valid_lost = 0
        valid_accuracy = 0

        for ii, (images2, labels2) in enumerate(vd_dl):
          optimizer.zero_grad()
          
          if use_gpu:
            images2, labels2 = images2.to('cuda') , labels2.to('cuda')
            model.to('cuda')

          with torch.no_grad():    
            outputs = model.forward(images2)
            valid_lost = criterion(outputs, labels2)
            ps = torch.exp(outputs).data
            equality = (labels2.data == ps.max(1)[1])
            valid_accuracy += equality.type_as(torch.FloatTensor()).mean()

        valid_lost = valid_lost / len(vd_dl)
        valid_accuracy = valid_accuracy / len(vd_dl)

        print(f'Epoch: {e+1}/{epochs}...',
              f'Loss: {running_loss/print_every:.4f}',
              f'Validation Lost {valid_lost:.4f}',
              f'Accuracy: {valid_accuracy:.4f}')

        running_loss = 0


def save_checkpoint(data_dir, model, save_dir, arch, hidden_units, dropout, learning_rate, epochs):
  train_data, _, _ = transform_img(data_dir)
  model.class_to_idx =  train_data.class_to_idx
  # model.cpu
  torch.save({
                'arch': arch,
                'hidden_units': hidden_units,
                'dropout': dropout,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'state_dict': model.state_dict(),
                'class_to_idx': model.class_to_idx
              }, save_dir
            )


def load_checkpoint(checkpoint, use_gpu):
  checkpoint = torch.load(checkpoint)
  learning_rate = checkpoint['learning_rate']
  hidden_units = checkpoint['hidden_units']
  dropout = checkpoint['dropout']
  arch = checkpoint['arch']
  model, _, _ = network_struct(arch, dropout, hidden_units, learning_rate, use_gpu)
  model.class_to_idx = checkpoint['class_to_idx']
  model.load_state_dict(checkpoint['state_dict'])

  return model


def process_image(image):
  ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
  returns an Numpy array '''
  
  proc_img = Image.open(image)
  
  prepoceess_img = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  pymodel_img = prepoceess_img(proc_img)
  
  return pymodel_img


def predict(input, model, top_k, use_gpu):
  ''' Predict the class (or classes) of an image using a trained deep learning model. '''

  if use_gpu:
    model.to('cuda')

  img_torch = process_image(input)
  img_torch = img_torch.unsqueeze_(0)
  img_torch = img_torch.float()

  with torch.no_grad():
    if use_gpu:
      output = model.forward(img_torch.cuda())

    output = model.forward(img_torch)

  probability = F.softmax(output.data, dim=1)

  return probability.topk(top_k)