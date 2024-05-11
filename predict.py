import numpy as np
import json
import argparse
import utils

ap = argparse.ArgumentParser(description = 'predict.py')
ap.add_argument('input', default = './flowers/test/1/image_06743.jpg', nargs = '?', action = 'store', type = str)
ap.add_argument('--dir', action = 'store', dest = 'data_dir', default = './flowers/')
ap.add_argument('checkpoint', default = './checkpoint.pth', nargs = '?', action = 'store', type = str)
ap.add_argument('--top_k', default = 5, dest = 'top_k', action = 'store', type = int)
ap.add_argument('--category_names', dest = 'category_names', action = 'store', default = 'cat_to_name.json')
ap.add_argument('--gpu', default = 'cpu', action = 'store', dest = 'gpu')

pa = ap.parse_args()
input = pa.input
top_k = pa.top_k
checkpoint = pa.checkpoint
use_gpu = False if pa.gpu == 'cpu' else True

def main():
  model = utils.load_checkpoint(checkpoint, use_gpu)
  with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)
  probabilities = utils.predict(input, model, top_k, use_gpu)
  labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
  probability = np.array(probabilities[0][0])
  i = 0
  while i < top_k:
    print(f'{labels[i]} with a probability of {probability[i]}')
    i += 1
  print('Predicting Completed!')
    
if __name__ == '__main__':
  main()