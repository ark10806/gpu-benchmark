import argparse

def init():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--min_batch_size', type=int, default=16)
  parser.add_argument('--max_batch_size', type=int, default=2048)
  parser.add_argument('--epochs', default=3, type=int)
  parser.add_argument('--batch_size', default=512, type=int)
  parser.add_argument('--n_workers', default=4, type=int)
  return parser.parse_args()