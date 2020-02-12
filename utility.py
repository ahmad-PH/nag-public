import os; import time
from fastai.callbacks import *
from fastai.basic_train import *

def to_ordinal(i):
  ones = i % 10
  if ones == 1: suffix = 'st'
  elif ones == 2: suffix = 'nd'
  elif ones == 3: suffix = 'rd'
  else: suffix = 'th'
  return str(i) + suffix

def get_shell_output(command):
    return subprocess.check_output(command, shell = True).decode("utf-8")

def run_shell_cmd(command):
    process = subprocess.Popen(command, shell = True, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
    ignore_out, ignore_err = process.communicate()
    return process.wait()


class FileControl(LearnerCallback):
    def __init__(self, learn, root_folder, gen):
      super().__init__(learn)
      self.root_folder = root_folder
      self.gen = gen

    def on_epoch_end(self, epoch, **kwargs):
      with open(self.root_folder + '/ctrl.txt', 'r') as control_file:
        control_data = json.loads(control_file.read())

      if str(epoch) in control_data:
        action = control_data[str(epoch)]
        self.epoch = epoch
        return self.perform_action(action)

    def perform_action(self, action):
      if action == 'stop':
        return {'stop_training': True}
      elif action == 'ask':
        print('prompted to ask for action at epoch {}:'.format(self.epoch))
        new_action = input()
        self.perform_action(new_action)
      elif action == 'ask_file':
        return self.ask_from_file()
      elif action == 'double_labels':
        self.gen.n_active_labels = min(self.gen.n_active_labels * 2, 1000)
        print('increased n_active_labels to {} at end of epoch {}'.format(self.gen.n_active_labels ,self.epoch))
      elif action == 'continue':
        return
      else:
        print('invalid action: \"{}\". please enter a valid action:'.format(action))
        return self.perform_action(input())

    def ask_from_file(self):
      wait_file = open(self.root_folder + '/wait.txt', 'w')
      while True:
        if not os.path.isfile(self.root_folder + '/answer.txt'):
          open(self.root_folder + '/answer.txt', 'x')
        answers_file = open(self.root_folder + '/answer.txt', 'r')
        action = answers_file.read().strip()
        if action in ['stop', 'double_labels', 'continue']:
          print('action read: \"{}\"'.format(action))
          wait_file.close()
          os.remove(self.root_folder + '/wait.txt')
          return self.perform_action(action)
        else:
          wait_file.truncate(0)
          wait_file.write('invalid action \"{}\"\n'.format(action))
          wait_file.flush()
          time.sleep(10)

def derange(*args):
  if len(args) == 0: raise ValueError('shuffle function needs atleast one argument')
  deranged_indexes = derangement(args[0].shape[0])
  if not all([args[0].shape[0] == arg.shape[0] for arg in args]): 
    raise ValueError('inputs to shuffle must all have the same 0th dimension')
  return [arg[deranged_indexes] for arg in args]