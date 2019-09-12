import torch
import torch.nn as nn
import torch.nn.functional as F

torch.Tensor.ndims = property(lambda x: len(x.shape))

__all__ = [
    'VBN',
]

def batch_normalization(x, mean, variance, offset, scale, variance_epsilon, name=None):
  inv = 1 / ((variance + variance_epsilon) ** 0.5)
  if scale is not None:
    inv *= scale
  if offset is None:
    return (x-mean) * inv 
  else:
    return (x-mean) * inv + offset

def _validate_init_input_and_get_axis(reference_batch, axis):
  """Validate input and return the used axis value."""
  if reference_batch.ndims is None:
    raise ValueError('`reference_batch` has unknown dimensions.')

  ndims = reference_batch.ndims
  if axis < 0:
    used_axis = ndims + axis
  else:
    used_axis = axis
  if used_axis < 0 or used_axis >= ndims:
    raise ValueError('Value of `axis` argument ' + str(used_axis) +
                     ' is out of range for input with rank ' + str(ndims))
  return used_axis


def _statistics(x, axes):
  """Calculate the mean and mean square of `x`.
  Modified from the implementation of `tf.nn.moments`.
  Args:
    x: A `Tensor`.
    axes: Array of ints.  Axes along which to compute mean and variance.
  Returns:
    Two `Tensor` objects: `mean` and `square mean`.
  """
  # The dynamic range of fp16 is too limited to support the collection of
  # sufficient statistics. As a workaround we simply perform the operations
  # on 32-bit floats before converting the mean and variance back to fp16
  y = x.type(torch.float32) if x.dtype == torch.float16 else x

  # Compute true mean while keeping the dims for proper broadcasting.
  with torch.no_grad():
    shift = torch.mean(y, axes, keepdim=True)

    shifted_mean = torch.mean(y - shift, axes, keepdim=True)
    mean = shifted_mean + shift
    mean_squared = torch.mean(y**2, axes, keepdim=True)

    mean = mean.squeeze()
    mean_squared = mean_squared.squeeze()
    if x.dtype == torch.float16:
      return (mean.type(torch.float16),
              mean_squared.type(torch.float16))
    else:
      return (mean, mean_squared)



def _validate_call_input(tensor_list, batch_dim):
  """Verifies that tensor shapes are compatible, except for `batch_dim`."""

  def _get_shape(tensor):
    shape = list(tensor.shape)
    del shape[batch_dim]
    return shape

  base_shape = _get_shape(tensor_list[0])
  for tensor in tensor_list:
    if base_shape != _get_shape(tensor):
      raise ValueError('in _validate_call_input, incompatible shapes: {}, {}'.format(
        tensor_list[0].shape, tensor.shape  
      ))

class VBN(nn.Module):

  def __init__(self,
               axis=1,
               epsilon=1e-3,
               center=True,
               scale=True,
               trainable=True,
               batch_axis=0):
    super().__init__()
    self._axis = axis
    self._epsilon = epsilon
    self._center = center
    self._scale = scale
    self._trainable = trainable
    self._batch_axis = batch_axis
    self._reference_batch = None

  def register_reference_batch(self,
               reference_batch):
    axis = _validate_init_input_and_get_axis(reference_batch, self._axis)
    self._batch_axis = _validate_init_input_and_get_axis(
        reference_batch, self._batch_axis)

    if axis == self._batch_axis:
      raise ValueError('`axis` and `batch_axis` cannot be the same.')

    self._reference_batch = reference_batch

    # Calculate important shapes:
    #  1) Reduction axes for the reference batch
    #  2) Broadcast shape, if necessary
    #  3) Reduction axes for the virtual batchnormed batch
    #  4) Shape for optional parameters
    input_shape = self._reference_batch.shape
    ndims = self._reference_batch.ndims
    reduction_axes = list(range(ndims))
    del reduction_axes[axis]

    self._broadcast_shape = [1] * len(input_shape)
    self._broadcast_shape[axis] = input_shape[axis]

    self._example_reduction_axes = list(range(ndims))
    del self._example_reduction_axes[max(axis, self._batch_axis)]
    del self._example_reduction_axes[min(axis, self._batch_axis)]

    params_shape = self._reference_batch.shape[axis]

    # Determines whether broadcasting is needed. This is slightly different
    # than in the `nn.batch_normalization` case, due to `batch_dim`.
    self._needs_broadcasting = (
        sorted(self._example_reduction_axes) != list(range(ndims))[:-2])

    # Calculate the sufficient statistics for the reference batch in a way
    # that can be easily modified by additional examples.
    self._ref_mean, self._ref_mean_squares = _statistics(
        self._reference_batch, reduction_axes)
    self._ref_variance = (
        self._ref_mean_squares - self._ref_mean ** 2)

    # Virtual batch normalization uses a weighted average between example
    # statistics and the reference batch statistics.
    ref_batch_size = self._reference_batch.shape[self._batch_axis]

    self._example_weight = 1. / (float(ref_batch_size) + 1.)
    self._ref_weight = 1. - self._example_weight

    # Make the variables, if necessary.
    if self._center:
      self._beta = nn.Parameter(torch.zeros(params_shape), self._trainable)
    if self._scale:
      self._gamma = nn.Parameter(torch.ones(params_shape), self._trainable)

  def _virtual_statistics(self, inputs, reduction_axes):
    """Compute the statistics needed for virtual batch normalization."""
    cur_mean, cur_mean_sq = _statistics(inputs, reduction_axes)
    vb_mean = (
        self._example_weight * cur_mean + self._ref_weight * self._ref_mean)
    vb_mean_sq = (
        self._example_weight * cur_mean_sq +
        self._ref_weight * self._ref_mean_squares)
    return (vb_mean, vb_mean_sq)

  def _broadcast(self, v, broadcast_shape=None):
    # The exact broadcast shape depends on the current batch, not the reference
    # batch, unless we're calculating the batch normalization of the reference
    # batch.
    b_shape = broadcast_shape or self._broadcast_shape
    if self._needs_broadcasting and v is not None:
      return v.view(b_shape)
    return v

  # def reference_batch_normalization(self):
  # """Return the reference batch, but batch normalized."""
  #   return F.batch_norm(self._reference_batch,
  #                       self._broadcast(self._ref_mean),
  #                       self._broadcast(self._ref_variance),
  #                       self._broadcast(self._gamma),
  #                       self._broadcast(self._beta),
  #                       #Training = ?
  #                       eps = self._epsilon)

  def forward(self, inputs):
    """Run virtual batch normalization on inputs.
    Args:
      inputs: Tensor input.
    Returns:
       A virtual batch normalized version of `inputs`.
    Raises:
       ValueError: If `inputs` shape isn't compatible with the reference batch.
    """
    if self._reference_batch is None:
      self.register_reference_batch(inputs)

    _validate_call_input([inputs, self._reference_batch], self._batch_axis)

    # Calculate the statistics on the current input on a per-example basis.
    vb_mean, vb_mean_sq = self._virtual_statistics(
        inputs, self._example_reduction_axes)
    vb_variance = vb_mean_sq - vb_mean ** 2

    # The exact broadcast shape of the input statistic Tensors depends on the
    # current batch, not the reference batch. The parameter broadcast shape
    # is independent of the shape of the input statistic Tensor dimensions.
    b_shape = self._broadcast_shape[:]  # deep copy
    b_shape[self._batch_axis] = inputs.shape[self._batch_axis]
    
    # print('inputs: {}, b_shape: {}, bc_shape: {}, vb_mean: {}, broadcast(vb_mean): {}, beta: {}, broadcast(beta): {}'.format(
    #   inputs.shape, b_shape, self._broadcast_shape, vb_mean.shape, self._broadcast(vb_mean, self._broadcast_shape).shape, 
    #   self._beta.shape, self._broadcast(self._beta, self._broadcast_shape).shape))

    return batch_normalization(
        inputs, 
        self._broadcast(vb_mean, b_shape),
        self._broadcast(vb_variance, b_shape),
        self._broadcast(self._beta, self._broadcast_shape),
        self._broadcast(self._gamma, self._broadcast_shape),
        self._epsilon)