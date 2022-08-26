import argparse

def init():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--min_batch_size', type=int, default=256)
  parser.add_argument('--max_batch_size', type=int, default=4096)
  parser.add_argument('--epochs', default=1, type=int)
  parser.add_argument('--batch_size', default=-1, type=int, help='-1: uses range(min, max)batchsizes / N: uses single batchsize')
  parser.add_argument('--n_workers', default=4, type=int, help='the number of threads that responsible for preprocess data')
  parser.add_argument('--n_gpu', default=1, type=int, help='-1 for max gpus')
  return parser.parse_args()