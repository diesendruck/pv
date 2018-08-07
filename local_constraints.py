# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=invalid-name

#THIS IS A LOCAL VERSION WITH ADAPTED NORMS
#   Note: Original files are here:
#     /usr/local/lib/python2.7/dist-packages/tensorflow/python/keras/_impl/keras 
#
#Constraints: functions that impose constraints on weight values.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import operator

from tensorflow.python.keras._impl.keras import backend as K
from tensorflow.python.keras._impl.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras._impl.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import tf_export


@tf_export('keras.constraints.Constraint')
class Constraint(object):

  def __call__(self, w):
    return w

  def get_config(self):
    return {}


@tf_export('keras.constraints.MaxNorm', 'keras.constraints.max_norm')
class MaxNorm(Constraint):
  """MaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have a norm less than or equal to a desired value.

  Arguments:
      m: the maximum norm for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  """

  def __init__(self, max_value=2, axis=0):
    self.max_value = max_value
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = K.clip(norms, 0, self.max_value)
    return w * (desired / (K.epsilon() + norms))

  def get_config(self):
    return {'max_value': self.max_value, 'axis': self.axis}


@tf_export('keras.constraints.NonNeg', 'keras.constraints.non_neg')
class NonNeg(Constraint):
  """Constrains the weights to be non-negative.
  """

  def __call__(self, w):
    return w * math_ops.cast(math_ops.greater_equal(w, 0.), K.floatx())


@tf_export('keras.constraints.UnitNorm', 'keras.constraints.unit_norm')
class UnitNorm(Constraint):
  """Constrains the weights incident to each hidden unit to have unit norm.

  Arguments:
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, w):
    return w / (
        K.epsilon() + K.sqrt(
            math_ops.reduce_sum(
                math_ops.square(w), axis=self.axis, keepdims=True)))

  def get_config(self):
    return {'axis': self.axis}


@tf_export('keras.constraints.MinMaxNorm', 'keras.constraints.min_max_norm')
class MinMaxNorm(Constraint):
  """MinMaxNorm weight constraint.

  Constrains the weights incident to each hidden unit
  to have the norm between a lower bound and an upper bound.

  Arguments:
      min_value: the minimum norm for the incoming weights.
      max_value: the maximum norm for the incoming weights.
      rate: rate for enforcing the constraint: weights will be
          rescaled to yield
          `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
          Effectively, this means that rate=1.0 stands for strict
          enforcement of the constraint, while rate<1.0 means that
          weights will be rescaled at each step to slowly move
          towards a value inside the desired interval.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  def __call__(self, w):
    norms = K.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * K.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    return w * (desired / (K.epsilon() + norms))

  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }


###############################################################################
# NEW NORMS

@tf_export('keras.constraints.DivideByMaxNorm', 'keras.constraints.divide_by_max_norm')
class DivideByMaxNorm(Constraint):
  """DivideByMaxNorm weight constraint.

  Divides all weights by max absolute values, so that max magnitude is -1 or 1.

  Arguments:
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  """

  def __init__(self, axis=0):
    self.axis = axis

  def __call__(self, w):
    maximum_weight = K.max(K.abs(w))
    return w / (K.epsilon() + maximum_weight)

  def get_config(self):
    return {'axis': self.axis}


@tf_export('keras.constraints.ClipNorm', 'keras.constraints.clip_norm')
class ClipNorm(Constraint):
  """ClipNorm weight constraint.

  Constrains the weights by clipping values to be on some interval.

  Arguments:
      min_value: the minimum value for the incoming weights.
      max_value: the maximum value for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  """

  def __init__(self, min_value=-1, max_value=1, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.axis = axis

  def __call__(self, w):
    return K.clip(w, self.min_value, self.max_value)

  def get_config(self):
      return {'min_value': self.min_value,
              'max_value': self.max_value,
              'axis': self.axis}


@tf_export('keras.constraints.EdgeIntervalNorm', 'keras.constraints.edge_interval_norm')
class EdgeIntervalNorm(Constraint):
  # TODO: Figure out. Idea is to push weights to the edge of the interval.
  """EdgeIntervalNorm weight constraint.

  Constrains the weights by scaling values to be on some outer hyperdisk. To do
  this, first norm as in DivideByMax, to put weights on [-1, 1]. Then extract
  signs, apply absolute value, scale [0,1] to [min_value,1], and reapply signs.

  Specifically, for min_value=0.8, after norming, extracting sign, and absolute
  value, left with weights on [0,1]. By multiplying by (1 - min_value), get 
  weights on [0,0.2]. Then adding min_value gives weights on [0.8,1].

  Arguments:
      min_value: the minimum value for the incoming weights.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.

  """

  def __init__(self, min_value=0, axis=0):
    self.min_value = min_value
    self.axis = axis

  #min_value = property(operator.attrgetter('_min_value'))
  #@min_value.setter
  #def min_value(self, v):
  #  if not (v >= 0 and v < 1): raise Exception('min_value must be on [0, 1)')
  #    self._min_value = v

  def __call__(self, w):
    maximum_weight = K.max(K.abs(w))
    w = w / (K.epsilon() + maximum_weight)  # On [-1,1].
    signs = K.sign(w)
    unsigned_w = K.abs(w)  # On [0,1].
    edge_scaled_w_unsigned = unsigned_w * (1. - self.min_value) + self.min_value  # On [min_value,1].
    edge_scaled_w = signs * edge_scaled_w_unsigned  # On [-1,-min_value] U [min_value,1].
    return edge_scaled_w

  def get_config(self):
      return {'min_value': self.min_value,
              'axis': self.axis}


@tf_export('keras.constraints.DivideByMaxThenMinMaxNorm', 'keras.constraints.divide_by_max_then_min_max_norm')
class DivideByMaxThenMinMaxNorm(Constraint):
  """DivideByMaxThenMinMaxNorm weight constraint.

  Constrains weights to [-1,1], then constrains norm of weights to interval.

  Arguments:
      min_value: the minimum norm for the incoming weights.
      max_value: the maximum norm for the incoming weights.
      rate: rate for enforcing the constraint: weights will be
          rescaled to yield
          `(1 - rate) * norm + rate * norm.clip(min_value, max_value)`.
          Effectively, this means that rate=1.0 stands for strict
          enforcement of the constraint, while rate<1.0 means that
          weights will be rescaled at each step to slowly move
          towards a value inside the desired interval.
      axis: integer, axis along which to calculate weight norms.
          For instance, in a `Dense` layer the weight matrix
          has shape `(input_dim, output_dim)`,
          set `axis` to `0` to constrain each weight vector
          of length `(input_dim,)`.
          In a `Conv2D` layer with `data_format="channels_last"`,
          the weight tensor has shape
          `(rows, cols, input_depth, output_depth)`,
          set `axis` to `[0, 1, 2]`
          to constrain the weights of each filter tensor of size
          `(rows, cols, input_depth)`.
  """

  def __init__(self, min_value=0.0, max_value=1.0, rate=1.0, axis=0):
    self.min_value = min_value
    self.max_value = max_value
    self.rate = rate
    self.axis = axis

  def __call__(self, w):
    # First apply DivideByMax.
    maximum_weight = K.max(K.abs(w))
    w = w / (K.epsilon() + maximum_weight)  # On [-1, 1].
    # Then apply MinMaxNorm.
    norms = K.sqrt(
        math_ops.reduce_sum(math_ops.square(w), axis=self.axis, keepdims=True))
    desired = (
        self.rate * K.clip(norms, self.min_value, self.max_value) +
        (1 - self.rate) * norms)
    return w * (desired / (K.epsilon() + norms))

  def get_config(self):
    return {
        'min_value': self.min_value,
        'max_value': self.max_value,
        'rate': self.rate,
        'axis': self.axis
    }

# Aliases.

max_norm = MaxNorm
non_neg = NonNeg
unit_norm = UnitNorm
min_max_norm = MinMaxNorm

# Legacy aliases.
maxnorm = max_norm
nonneg = non_neg
unitnorm = unit_norm


@tf_export('keras.constraints.serialize')
def serialize(constraint):
  return serialize_keras_object(constraint)


@tf_export('keras.constraints.deserialize')
def deserialize(config, custom_objects=None):
  return deserialize_keras_object(
      config,
      module_objects=globals(),
      custom_objects=custom_objects,
      printable_module_name='constraint')


@tf_export('keras.constraints.get')
def get(identifier):
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, six.string_types):
    config = {'class_name': str(identifier), 'config': {}}
    return deserialize(config)
  elif callable(identifier):
    return identifier
  else:
    raise ValueError('Could not interpret constraint identifier: ' +
                     str(identifier))
