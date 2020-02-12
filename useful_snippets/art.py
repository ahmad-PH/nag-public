# It comes right after creating the `denoiser_learn` object

# class ArtWrapper(nn.Module):
#   def __init__(self, wrap_around):
#     super().__init__()
#     self.wrap_around = wrap_around
    
#   def forward(self, inp):
# #     print('passed input:', inp.shape)
# #     print('converted: ', inp[None].shape)
#     out = self.wrap_around(inp[None])
# #     print('out shape: ', out[0].shape)
#     final_out = arch(out[0]).squeeze()
# #     print('final out shape: ', final_out.shape)
#     return final_out

# class BasicClassifierLoss(nn.Module):
#   def __init__(self):
#       super().__init__()
    
#   def forward(self, inp, target):
# #     return -1 * torch.log(torch.gather(inp, 1, target.unsqueeze(0)))
#     return -1 * torch.log(torch.gather(inp, 0, target))




# from art import metrics
# from art.classifiers import PyTorchClassifier

# x =  denoiser_learn.data.train_ds[0][0].data.numpy()
# y =  denoiser_learn.data.train_ds[0][1]
# one_hot = [0.] * 1000
# one_hot[363] = 1.
# y = np.array(one_hot)
# print(x.shape, y.shape)

# wrapped_model = ArtWrapper(denoiser_learn.model)
# # WARNING: omitting clip values
# art_classifier = PyTorchClassifier(model = wrapped_model, loss = BasicClassifierLoss(), #feat_loss
#                                    optimizer = denoiser_learn.opt, input_shape = x.shape,
#                                    nb_classes = 1000)

# # metrics.empirical_robustness(art_classifier, x, 'fgsm')
# metrics.loss_sensitivity(art_classifier, x, y)