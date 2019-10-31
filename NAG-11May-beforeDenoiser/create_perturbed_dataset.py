from nag_util import print_range
import nag_util
import os


def perturb_dataset(trainOrValid: str, model: nn.Module, dest_folder: str):

  data = (ImageList.from_folder(env.data_path)
          .split_by_folder(valid='valid')
          .label_from_folder()
          .transform(None, size=224, resize_method=ResizeMethod.SQUISH)
          .databunch(bs=32, num_workers=1)
          .normalize(imagenet_stats))

  if trainOrValid == 'train':
    dataloader = data.train_dl
  elif trainOrValid == 'valid':
    dataloader = data.valid_dl
  else:
    raise ValueError('invalid value for trainOrValid')

  for i in range(1000):
    os.makedirs("{}/{}".format(dest_folder, i))

  model = model.eval()
  next_filename_per_label = [0] * 1000

  with torch.no_grad():
    for batch, target_labels in dataloader:
      perturbation = model(batch)[0]
      perturbed_batch = batch + perturbation
      # The image is denormalized with imagenet stats from the range [-2.5, 2.5] to the range [0.,1.].
      denormalized_batch = torch.clamp(nag_util.denormalize(perturbed_batch), 0., 1.)
      for i, image_data in enumerate(denormalized_batch):
        image = Image(image_data)
        target_label = target_labels[i].item()
        print(target_labels[i].item())
        image.save("{}/{}/{}.jpg".format(dest_folder, target_label, next_filename_per_label[target_label]))
        next_filename_per_label[target_label] += 1
        
# code to test a single image:

# import PIL
# from torchvision import transforms

# with torch.no_grad():
#   img = PIL.Image.open('/home/mlcm-deep/AhmadPourihosseini/NAG/datasets/dataset/train/n01443537/6.jpg')
#   t = transforms.ToTensor()(transforms.Resize((224,224))(img)).cuda()
#   t = nag_util.normalize(t)
#   print_range(t)
#   p = model.generate_single_noise()
#   print_range(p)
#   perturbed_t = t + p
#   print(perturbed_t.shape)
#   denorm = torch.clamp(nag_util.denormalize(perturbed_t), 0., 1.)
#   print_range(denorm)
#   i = Image(denorm)
#   i.save('/home/mlcm-deep/AhmadPourihosseini/NAG/test.jpg')


# example of calling it:
# perturb_dataset(transform_data.train_dl, gen_learn.model, '/home/mlcm-deep/AhmadPourihosseini/NAG/transformed')