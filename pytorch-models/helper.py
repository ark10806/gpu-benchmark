from time import time

class TPS:
  def __init__(self):
    self.iter = 0
    self.time = time()
  
  def append(self, n):
    self.iter += n
  
  def eval(self):
    return str(self.iter / (time() - self.time))