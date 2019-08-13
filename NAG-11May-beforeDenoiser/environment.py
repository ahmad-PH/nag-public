import os
import subprocess

def detect_env():
    return 'colab' if 'content' in os.listdir('/') else 'IBM'
  
def run_shell_command(cmd):
  p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
  print(str(p.communicate()[0], 'utf-8'))
  
def create_env():
  if detect_env() == "IBM":
    return IBMEnv()
  elif detect_env() == "colab":
    return ColabEnv()

class Env:
  def get_csv_path(self):
    return self.get_csv_dir() + self.save_filename
  
  def get_models_path(self):
    return self.get_models_dir() + self.save_filename
  

class IBMEnv(Env):
    def __init__(self):
      self.root_folder = "/root/Derakhshani/adversarial"
      self.temp_csv_path = self.root_folder + "/temp"
      self.python_files_path = self.root_folder + "/nag-public"
      
      import sys
      sys.path.append('./nag/nag_util')
    
    def get_csv_dir(self):
      return self.root_folder + "/textual_notes/CSVs/"
    
    def get_models_dir(self):
      return self.root_folder + "/models/"
    
    def setup(self):
      import os; import torch;
      os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
      cuda_index = 0
      os.environ['CUDA_VISIBLE_DEVICES']=str(cuda_index)
      
    def load_dataset(self, compressed_name, unpacked_name):
      pass

    def load_test_dataset(self, root_folder):
      pass
    
    def set_data_path(self, path):
      self.data_path = Path(self.root_folder + '/datasets/' + path)
    
        
class ColabEnv(Env):
    def __init__(self):
      self.root_folder = '/content'
      self.temp_csv_path = self.root_folder
      self.python_files_path = self.root_folder + '/nag-public'
      self.torchvision_upgraded = False
      
    def get_csv_dir(self):
      return self.root_folder + '/gdrive/My Drive/DL/textual_notes/CSVs/'
    
    def get_models_dir(self):
      return self.root_folder + '/gdrive/My Drive/DL/models/'
        
    def setup(self):
        raise NotImplementedError('setup funtion has not been tested with the new run_shell_command yet.')
        # remove this once tochvision 0.3 is present by default in colab
        global torchvision_upgraded
        try:
            torchvision_upgraded
        except NameError:
          run_shell_command('pip uninstall -y torchvision')
          run_shell_command('pip install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp36-cp36m-linux_x86_64.whl')
          torchvision_upgraded = True
        else:
          print("torchvision already upgraded")
          
        from google.colab import drive
        drive.mount('/content/gdrive')
        
    def load_dataset(self, compressed_name, unpacked_name):
      raise NotImplementedError('load_dataset for colab has direct shell commands, which have not been tested yet.')
      if compressed_name not in os.listdir('.'):
        print(compressed_name + ' not found, getting it from drive')
        shutil.copyfile("/content/gdrive/My Drive/DL/{}.tar.gz".format(compressed_name), "./{}.tar.gz".format(compressed_name))

        gunzip_arg = "./{}.tar.gz".format(compressed_name)
        subprocess.call('gunzip -f ' + gunzip_arg) # NOT TESTED. original:  !gunzip -f $gunzip_arg

        tar_arg = "./{}.tar".format(compressed_name)
        subprocess.call('tar -xvf ' + tar_arg + ' > /dev/null') # NOTE TESTED. original: !tar -xvf $tar_arg > /dev/null

        os.rename(unpacked_name, compressed_name)
        subprocess.call('rm ' + tar_arg) # NOT TESTED. original: !rm $tar_arg
        
        print("done") 
      else:
        print(compressed_name + " found")
        
    def load_test_dataset(self, root_folder):
      test_folder = root_folder + '/test/'
      if 'test' not in os.listdir(root_folder):
        print('getting test dataset from drive')
        os.mkdir(test_folder)
        for i in range(1,11):
          shutil.copy("/content/gdrive/My Drive/DL/full_test_folder/{}.zip".format(i), test_folder)
          shutil.unpack_archive(test_folder + "/{}.zip".format(i), test_folder)
          os.remove(test_folder + "/{}.zip".format(i))
          print("done with the {}th fragment".format(i))
      else:
        print('test dataset found.')
        
    def set_data_path(self, path):
      self.data_path = Path('./' + path)
        