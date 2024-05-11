import argparse
import utils

ap = argparse.ArgumentParser(description = 'train.py')
ap.add_argument('data_dir', action = 'store', default = './flowers/')
ap.add_argument('--gpu', dest = 'gpu', action = 'store', default = 'cpu')
ap.add_argument('--save_dir', dest = 'save_dir', action = 'store', default = './checkpoint.pth')
ap.add_argument('--learning_rate', dest = 'learning_rate', action = 'store', default = 0.001)
ap.add_argument('--dropout', dest = 'dropout', action = 'store', default = 0.5)
ap.add_argument('--epochs', dest = 'epochs', action = 'store', type = int, default = 10)
ap.add_argument('--arch', dest = 'arch', action = 'store', default = 'vgg16', type = str)
ap.add_argument('--hidden_units', type = int, dest = 'hidden_units', action = 'store', default = 256)

pa = ap.parse_args()
data_dir = pa.data_dir
save_dir = pa.save_dir
learning_rate = pa.learning_rate
arch = pa.arch
dropout = pa.dropout
hidden_units = pa.hidden_units
epochs = pa.epochs
print_every = 20
use_gpu = False if pa.gpu == 'cpu' else True

def main():
  trainloader, _, _ = utils.load_data(data_dir)
  model, optimizer, criterion = utils.network_struct(arch, dropout, hidden_units, learning_rate, use_gpu)
  utils.train_neural_net(data_dir, model, criterion, optimizer, trainloader, epochs, print_every, use_gpu)
  utils.save_checkpoint(data_dir, model, save_dir, arch, hidden_units, dropout, learning_rate, epochs)
  print('Training Completed!')

if __name__ == '__main__':
  main()