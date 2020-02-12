from fastai.vision import *
from fastai.imports import *
from fastai.callbacks import *
from fastai.utils.mem import *

class FoolingWeightScheduler(LearnerCallback):
  def __init__(self, learn: Learner):
    super().__init__(learn)
    self.weights_history = []
    self.fooling_loss_history = []
  
  def get_metric_value(self, metric_name):
    for value, name in zip(self.learn.recorder.metrics[-1],self.learn.recorder.names[3:-1]):
      if name == metric_name:
        return value
    raise ValueError('Could not find {} metric.'.format(metric_name))
  
  def on_epoch_end(self, last_metrics, **kwargs):
    # history keeping
    self.weights_history.append((kwargs['epoch'], self.learn.loss_func.fooling_weight))
    
    # the actual functionality
    fooling_loss = self.get_metric_value('fool_loss')
    self.fooling_loss_history.append(fooling_loss)
    
    if len(self.weights_history) < 2:
      return
    
    if self.fooling_loss_history[-1] > self.fooling_loss_history[-2]:
      self.learn.loss_func.fooling_weight += 0.3    
      print('fooling weight increased to {} at the end of epoch {}'.format(
        self.learn.loss_func.fooling_weight, kwargs['epoch']))


class CyclicalLRScheduler(LearnerCallback):
  def __init__(self, learn, max_lr, min_lr, cycle_len):
    super().__init__(learn)
    self.max_lr = max_lr
    self.min_lr = min_lr
    self.cycle_len = cycle_len
    
  def on_train_begin(self, **kwargs):
    self.n_iter_per_epoch = len(self.learn.data.train_dl)
    self.cycle_len_iters = self.cycle_len * self.n_iter_per_epoch
    self.learn.opt.lr = self.min_lr
    
    
  def on_batch_end(self, iteration, train, **kwargs):
    if train:
      cycle_index = iteration % self.cycle_len_iters
      half_cycle_len = self.cycle_len_iters / 2

      if cycle_index < half_cycle_len:
        new_lr = float(self.max_lr - self.min_lr) / half_cycle_len * cycle_index + self.min_lr
      else:
        new_lr = float(self.min_lr - self.max_lr) / half_cycle_len * (cycle_index - half_cycle_len) + self.max_lr

#       print('iter: {}, lr: {}'.format(iteration, new_lr))
      self.opt.lr = new_lr