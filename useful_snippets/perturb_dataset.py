# It comes right after creating the main `data` object 


# data_perturbed = (ImageList.from_folder('/home/mlcm-deep/AhmadPourihosseini/NAG/datasets/perturbed_resnet50_72')
#         .split_by_folder(valid=('test' if mode == 'div_metric_calc' else 'valid'))
#         .label_from_folder()
#         .transform(None, size=224)
#         .databunch(bs=batch_size, num_workers=1)
#         .normalize(imagenet_stats))



# # TO CONTINUE:
# import os

# def denoiser_label_func(x):
#   dataset_type = x.parts[-3]
#   class_name = x.parts[-2]
#   img_name = os.path.splitext(x.parts[-1])[0]
#   return os.path.join(*x.parts[:-4], 'dataset', dataset_type, class_name, img_name + '.jpg')

# data_denoiser = (ImageImageList.from_folder('/home/mlcm-deep/AhmadPourihosseini/NAG/datasets/perturbed_resnet50_72')
#         .split_by_folder(valid=('test' if mode == 'div_metric_calc' else 'valid'))
#         .label_from_func(denoiser_label_func)
#         .transform(None, size=224, tfm_y=True)
#         .databunch(bs=batch_size, num_workers=1)
#         .normalize(imagenet_stats))




# j = 0
# for i, (d, t) in enumerate(data.fix_dl):
#   print(data.fix_ds.items[i], t[0])
#   Image(nag_util.denormalize(d)).show()
#   j += 1
#   if j == 20:
#     break
  
# # for d, t in data_denoiser.train_dl:
# #   im1 = Image(d[0])
# #   im2 = Image(t[0])
# #   im1.show(), im2.show()
# #   break



# # test site:
# l = Learner(data_perturbed, model(True), metrics = [accuracy])
# l.validate(data_perturbed.valid_dl)




# l = Learner(data, model(True), metrics = [accuracy])
# l.validate(data.valid_dl)