from fastai.vision import *
from fastai.imports import *
from fastai.callbacks import *
from fastai.utils.mem import *
import torch
import torch.nn.functional as F
import torchvision
import os
import subprocess

class deconv_layer(nn.Module):
    def __init__(self, in_ch, out_ch, k_size = (4,4), s = (2,2), pad = (1,1), b = True, activation = True):
        super(deconv_layer, self).__init__()

        self.CT2d = nn.ConvTranspose2d(in_channels = in_ch,
                                  out_channels = out_ch,
                                  kernel_size = k_size,
                                  stride = s, 
                                  padding = pad,
                                  bias = b)
        self.BN2d = nn.BatchNorm2d(out_ch)
        self.activation = activation

        self.weight_init()
    
    def forward(self, input):
        if self.activation:
            return F.relu(self.BN2d(self.CT2d(input)), inplace=True)
        else:
            return self.BN2d(self.CT2d(input))

    def weight_init(self):
        self.CT2d.weight.data.normal_(mean = 0, std = 0.02)
        self.CT2d.bias.data.fill_(0)


def diversity_loss(input, target):
    return torch.mean(torch.nn.functional.cosine_similarity(
      input.view([batch_size, -1]),
      target.view([batch_size, -1]), 
    ))

  
global_perturbations = None

def validation_single_perturbation(gen_output, target):
  _, clean_images = gen_output
  perturbed_images = clean_images + global_perturbations
  
  benign_preds = torch.argmax(arch(clean_images), 1)
  adversary_preds = torch.argmax(arch(perturbed_images), 1)
  return (benign_preds != adversary_preds).float().mean()


def validation(gen_output, target):
	perturbations, _, _, clean_images = gen_output
	perturbed_images = clean_images + perturbations
	benign_preds = torch.argmax(arch(clean_images), 1)
	adversary_preds = torch.argmax(arch(perturbed_images), 1)
	return (benign_preds != adversary_preds).float().mean()


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0,2,1) # B X N X C
        proj_key =  self.key_conv(x).view(m_batchsize, -1, width*height) # B X C x (N)
        energy =  torch.bmm(proj_query, proj_key) # transpose check
        attention = self.softmax(energy) # B X (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height) # B X C X N

        out = torch.bmm(proj_value, attention.permute(0,2,1) )
        out = out.view(m_batchsize, C, width, height)
        
        out = self.gamma * out + x
        return out, attention

class AttnBasedGen(nn.Module):
    def __init__(self, z_dim, gf_dim=64, y_dim = None, df_dim = 64, image_shape = [3,128,128]):
        super(Gen, self).__init__()

        self.bs = None
        self.z_dim = z_dim
        self.gf_dim = gf_dim
        self.y_dim = y_dim
        self.df_dim = df_dim
        self.image_shape = image_shape

        self.z_ = nn.Linear(self.z_dim, self.gf_dim * 7 * 4 * 4, bias=True)
        self.z_.bias.data.fill_(0)
        self.BN_ = nn.BatchNorm2d(self.gf_dim * 7)

        self.CT2d_1 = deconv_layer(self.gf_dim * 8, 
                                 self.gf_dim * 4,
                                  k_size = (5,5), s = (2,2), pad = (2,2))
        self.CT2d_2 = deconv_layer(self.gf_dim * 5, self.gf_dim * 2)

        self.half = self.gf_dim // 2
        if self.half == 0:
          self.half == 1
        self.CT2d_3 = deconv_layer(self.gf_dim * 2 + self.half, self.gf_dim * 1)

        self.quarter = self.gf_dim // 4
        if self.quarter == 0:
          self.quarter == 1
        self.CT2d_4 = deconv_layer(self.gf_dim * 1 + self.quarter, self.gf_dim * 1)

        self.eighth = self.gf_dim // 8
        if self.eighth == 0:
          self.eighth == 1
        self.CT2d_5 = deconv_layer(self.gf_dim * 1 + self.eighth, self.gf_dim * 1)
        self.atten5 = Self_Attn(self.gf_dim * 1, 'relu')

        # sixteenth = self.gf_dim // 16
        # if half == 0:
          # half == 1
        self.CT2d_6 = deconv_layer(self.gf_dim * 1 + self.eighth, self.gf_dim * 1)
        self.atten6 = Self_Attn(self.gf_dim * 1, 'relu')

        # sixteenth = self.gf_dim // 16
        # if half == 0:
          # half == 1
        self.CT2d_7 = deconv_layer(self.gf_dim * 1 + self.eighth, 3, k_size = (5,5), s = (1,1), pad = (2,2), activation = False)


    def forward_specific_z(self, inputs, z):
        self.bs = inputs.shape[0]
         # define generator here
        # input: bs * 100
        # Linear (z_dim, gf_dim * 7 * 4 * 4), bias = (True, init with zero), 
        # Reshape (bs, gf_dim * 7 * 4 * 4) -> (bs, gf_dim * 7, 4 , 4)
        # Virtual Batch Norm = VBN
        # ReLU
        # h0 <- relu output
        h0 = F.relu(self.BN_(self.z_(z).contiguous().view(self.bs, -1, 4, 4)))
        assert h0.shape[2:] == (4, 4), "Non-expected shape, it shoud be (4,4)"

        # h0z = self.make_z([bs, gf_dim, 4, 4])
        # h0 = torch.cat([h0, h0z], dim=1)
        # h1 = deconv(gf_dim * 8, gf_dim * 4, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h1 = ReLU(VBN(h1))
        h0z = self.make_z([self.bs, self.gf_dim, 4, 4])
        h0 = torch.cat([h0, h0z], dim=1)
        h1 = self.CT2d_1(h0)
        assert h1.shape[2:] == (7, 7), "Non-expected shape, it shoud be (7,7)"

        # h1z = self.make_z([bs, gf_dim, 7, 7])
        # h1 = torch.cat([h1, h1z], dim=1)
        # h2 = deconv(gf_dim * 5, gf_dim * 2, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h2 = ReLU(VBN(h2))
        # assert output size (14,14)
        h1z = self.make_z([self.bs, self.gf_dim, 7, 7])
        h1 = torch.cat([h1, h1z], dim=1)
        h2 = self.CT2d_2(h1)
        assert h2.shape[2:] == (14,14), "Non-expected shape, it shoud be (14,14)"

        # h2z = self.make_z([bs, half, 14, 14])
        # h2 = torch.cat([h2, h2z], dim=1)
        # h3 = deconv(gf_dim  2 + half, gf_dim  1, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h3 = ReLU(VBN(h3))
        h2z = self.make_z([self.bs, self.half, 14, 14])
        h2 = torch.cat([h2, h2z], dim=1)
        h3 = self.CT2d_3(h2)
        assert h3.shape[2:] == (28,28), "Non-expected shape, it shoud be (28,28)"

        # h3z = self.make_z([bs, quarter, 28, 28])
        # h3 = torch.cat([h3, h3z], dim=1)
        # h4 = deconv(gf_dim * 1 + quarter, gf_dim * 1, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h4 = ReLU(VBN(h4))
        h3z = self.make_z([self.bs, self.quarter, 28, 28])
        h3 = torch.cat([h3, h3z], dim=1)
        h4 = self.CT2d_4(h3)
        assert h4.shape[2:] == (56,56), "Non-expected shape, it shoud be (56,56)"

        # h4z = self.make_z([bs, self.eighth, 56, 56])
        # h4 = torch.cat([h4, h4z], dim=1)
        # h5 = deconv(gf_dim * 1 + eighth, gf_dim * 1, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h5 = ReLU(VBN(h5))

        h4z = self.make_z([self.bs, self.eighth, 56, 56])
        h4 = torch.cat([h4, h4z], dim=1)
        h5 = self.CT2d_5(h4)
        assert h5.shape[2:] == (112,112), "Non-expected shape, it shoud be (112,112)"
        h5 = self.atten5(h5)

        # h5z = self.make_z([bs, eighth, 112, 112])
        # h5 = torch.cat([h5, h5z], dim=1)
        # h6 = deconv(gf_dim * 1 + eighth, gf_dim * 1, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h6 = ReLU(VBN(h5))
        h5z = self.make_z([self.bs, self.eighth, 112, 112])
        h5 = torch.cat([h5, h5z], dim=1)
        h6 = self.CT2d_6(h5)
        assert h6.shape[2:] == (224,224), "Non-expected shape, it shoud be (224,224)"
        h5 = self.atten5(h5)

        # h6z = self.make_z([bs, eighth, 224, 224])
        # h6 = torch.cat([h6, h6z], dim=1)
        # h7 = deconv(gf_dim * 1 + eighth, 3, kernel = (5, 5), stride = (2,2), padding = (2,2), bias = (True, 0))
        # h7 = ReLU(VBN(h7))
        h6z = self.make_z([self.bs, self.eighth, 224, 224])
        h6 = torch.cat([h6, h6z], dim=1)
        h7 = self.CT2d_7(h6)
        assert h7.shape[2:] == (224,224), "Non-expected shape, it shoud be (448,448)"

        # out = 10*tanh(h7)

        #     return 10 *F.tanh(h7)
        ksi = 10.0
        output_coeff = ksi / (255.0 * np.mean(imagenet_stats[1])) 
        # this coeff scales the output to be appropriate for images that are 
        # normalized using imagenet_stats (and are hence in the approximate [-2.5, 2.5]
        # interval)
        return output_coeff * torch.tanh(h7), inputs
        #     return 0.15 * torch.tanh(h7)

    def forward(self, inputs):
        self.bs = inputs.shape[0]
        z = inputs.new_empty([self.bs, self.z_dim]).uniform_(-1,1)
        return self.forward_specific_z(inputs, z)
       
  
    def make_z(self, in_shape):
        result = torch.empty(in_shape).uniform_(-1,1).cuda()
        return self.move_gpu(result)

    def move_gpu(self, inp):
        if gpu_flag:
            return inp.cuda()
        else:
            return inp



   
def rename_imagenet_folders(root_folder):
    name_map = {}
    with open('/content/imagenet_labels.txt', 'r') as f:
        for line in f:
            name_id, cls_id, name = line.strip().split(' ')
            name_map[int(cls_id)] = name_id

    for idx, name in name_map.items():
      if Path('{}/{}'.format(root_folder, idx)).exists():
        os.rename('{}/{}'.format(root_folder, idx), '{}/{}'.format(root_folder, name))


def zip_test_dataset(root_folder):
  test_folder = root_folder + '/test'
  if 'test' not in os.listdir(root_folder):
    for i in range(1,11):
      os.mkdir(test_folder)
      shutil.copy("/content/gdrive/My Drive/DL/full_test_folder/{}.zip".format(i), test_folder)
      shutil.unpack_archive(test_folder + "/{}.zip".format(i), test_folder)
      os.remove(test_folder + "/{}.zip".format(i))
      rename_imagenet_folders(test_folder)
      shutil.make_archive("/content/{}".format(i) , "zip", test_folder)
      shutil.rmtree(test_folder)
#       shutil.copy("/content/{}.zip".format(i), "/content/gdrive/My Drive/DL/full_test_folder")
      print("done with the {}th fragment".format(i))


def derangement(n):
    result = torch.randperm(n).tolist()

    for i in range(len(result)):
        if i == result[i]:
            ip = (1 if i == 0 else 0)
            result[i], result[ip] = result[ip], result[i]

    return torch.tensor(result)


def load_starting_point(learn, name, z_dim):
  if detect_env() != "colab":
    raise NotImplementedError("load_starting_point not implemented for non-colab environments yet.")
  import os
  identity_token = name + '-zdim' + str(z_dim)
  address = '/content/gdrive/My Drive/DL/model_starting_points/' + identity_token
  starting_point_exists = os.path.isfile(address + '.pth')
  if not starting_point_exists:
    print("\n\nno starting point found for model:" + identity_token + ". creating one from the current learner.\n\n")
    learn.save(address)
  learn.load(address)

def random_seed(seed_value, use_cuda):
    random.seed(seed_value) # python
    np.random.seed(seed_value) # numpy
    torch.manual_seed(seed_value) # pytorch
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False



class DiversityWeightsScheduler(LearnerCallback):
  def __init__(self, learn: Learner):
    super().__init__(learn)
    self.weight = None
  
  def on_epoch_end(self, **kwargs):
    last_metrics = kwargs['last_metrics']
    if len(last_metrics) == 1: #it's the lr_find() calling
      return
    
    validation = last_metrics[1].item()
    
    div_loss = last_metrics[3].item()
#     div_loss = (last_metrics[3].item() + last_metrics[4].item()) /2.
    
    if div_loss < 0.6:
      if self.weight != 0.5:
        print("end of epoch {} switching weights to 0.5".format(kwargs['epoch']))
      self.weight = 0.5
    if div_loss < 0.4:
      if self.weight != 0.1:
        print("end of epoch {} switching weights to 0.1".format(kwargs['epoch']))
      self.weight = 0.1
    else:
      if self.weight != 1.:
        print("end of epoch {} switching weights to 1".format(kwargs['epoch']))
      self.weight = 1.
      
    self.learn.loss_func.weights = [self.weight] * len(layers)


class ImmediateCSVLogger(CSVLogger):
  def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
      super().on_epoch_end(epoch, smooth_loss, last_metrics)
      self.file.flush() 


def print_range(x):
  print(x.min().item(), x.max().item())


def denormalize(tensor, stats = imagenet_stats):
  assert(tensor.dim() == 3 or tensor.dim() == 4)
  mean = torch.tensor(stats[0]).cuda()
  stddev = torch.tensor(stats[1]).cuda()
  if tensor.dim() == 3:
    return (tensor * stddev[:,None, None]) + mean[:, None, None]    
  elif tensor.dim() == 4:
    return (tensor * stddev[None, :, None, None]) + mean[None, :, None, None]   
  
def normalize(tensor, stats = imagenet_stats):
  assert(tensor.dim() == 3 or tensor.dim() == 4)
  mean = torch.tensor(stats[0]).cuda()
  stddev = torch.tensor(stats[1]).cuda()
  if tensor.dim() == 3:
    return (tensor - mean[:, None, None]) / stddev[:,None, None]    
  elif tensor.dim() == 4:
    return (tensor - mean[None, :, None, None]) / stddev[None, :, None, None]


def scale_to_range(tensor, _range):
  new_range_length = _range[1] - _range[0]
  old_range_length = tensor.max() - tensor.min()
  return ((tensor - tensor.min()) * new_range_length / old_range_length) + _range[0]


def noise_to_image(noise):
  return Image(scale_to_range(noise.detach(), (0.,1.)))


def interpolate(x1, x2, step):
  import math
  with torch.no_grad():
    delta = (x2 - x1) * step
    result = [x1]
    n = math.floor(1 / step) - 1
    for i in range(n):
      result.append(result[-1] + delta)
    result.append(x2)
    return result


def class_index_to_label(index):
    file = open(os.path.dirname(os.path.abspath(__file__)) + '/imagenet_clsidx_to_labels.txt')
    line = None
    for i in range(index + 1):
        line = file.readline()
    start = line.find(':') + 2
    label = line.strip('\n')[start:-2]
    return (index, label)

def entropy(x):
  if not isinstance(x, torch.Tensor):
    x = torch.tensor(x)
  x = x / torch.sum(x)
  epsilon = 1e-10
  return - torch.sum(x * (torch.log(x + epsilon) / torch.log(torch.tensor(2.))))


def big_vector_to_str(x, thresh = 0.01):
  torch.set_printoptions(precision=2, sci_mode=False, threshold=5000)  
  result = "["
  for i, x_i in enumerate(x.data):
    if abs(x_i) > thresh:
      result += "{}: {:.2f}".format(i, x_i.item()) 
      result += ", " if (i < x.shape[0]-1) else ""
  result += "]"
  return result

def print_big_vector(x, thresh = 0.01):
  print(big_vector_to_str(x, thresh))

def tensorify(x):
  return x if isinstance(x, torch.Tensor) else torch.tensor(x)


class SoftmaxWrapper(nn.Module):
  def __init__(self, m):
    super().__init__()
    self.m = m
    self.softmax = nn.Softmax(dim=-1)
    
  def forward(self, inp):
    out = self.m(inp)
    return self.softmax(out)
 

def compute_mean_prediction_histogram_slow(learn, perturbations):
  pred_histogram = torch.tensor([0] * 1000).detach_()
  for j, perturbation in enumerate(perturbations):
    pred_histogram_j = torch.tensor(compute_prediction_histogram(learn, perturbation, True)).detach_()
    pred_histogram += pred_histogram_j
    print("finished creating histogram for the {}th perturbation".format(j))
  
  pred_histogram = pred_histogram.float() / len(perturbations)
  return pred_histogram.tolist()


def diversity_slow(learn, n_perturbations, percentage = 95, p = None):
  pred_histogram = compute_mean_prediction_histogram(
      learn, generate_perturbations(learn, n_perturbations)
  )
  print("finished creating the prediction histogram")

  return classes_needed_to_reach(95, pred_histogram)

class DiversityMetricOld(Callback):
  def __init__(self, n_perturbations, percentage):
    super().__init__()
    self.name = "div_metric_old"
    self.n_perturbations = n_perturbations
    self.percentage = percentage
    self.learn = None
  
  def set_learner(self, learn):
    self.learn = learn
    
  def on_epoch_begin(self, **kwargs):
    global learn
    self.perturbations = generate_perturbations(self.learn, self.n_perturbations)
    self.pred_hist = torch.tensor([0] * 1000).detach_()
    
  def on_batch_end(self, last_output, last_target, **kwargs):
    images = last_output[3]
    for j, perturbation in enumerate(self.perturbations):
      perturbed_batch = images + perturbation[None]
      preds = arch(perturbed_batch).argmax(1)
      for pred in preds:
        self.pred_hist[pred] += 1
  
  def on_epoch_end(self, last_metrics, **kwargs):
    self.pred_hist = (self.pred_hist.float() / len(self.perturbations)).tolist()
    div_metric = classes_needed_to_reach(self.percentage, self.pred_hist)[0]
    return add_metrics(last_metrics, div_metric)