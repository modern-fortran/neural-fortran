module nf_layer_constructors

  !! This module provides the functions to instantiate specific layers.

  use nf_layer, only: layer
  use nf_activation, only : activation_function

  implicit none

  private
  public :: &
    conv, &
    dense, &
    dropout, &
    flatten, &
    input, &
    linear2d, &
    locally_connected1d, &
    maxpool, &
    reshape, &
    self_attention, &
    embedding, &
    layernorm

  interface input

    module function input1d(layer_size) result(res)
      !! 1-d input layer constructor.
      !!
      !! This layer is for inputting 1-d data to the network.
      !! Currently, this layer must be followed by a dense layer.
      !! An input layer must be the first layer in the network.
      !!
      !! This is a specific function that is available
      !! under a generic name `input`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: input, layer
      !! type(layer) :: input_layer
      !! input_layer = input(768)
      !! ```
      integer, intent(in) :: layer_size
        !! Size of the input layer
      type(layer) :: res
        !! Resulting layer instance
    end function input1d

    module function input2d(dim1, dim2) result(res)
      !! 2-d input layer constructor.
      !!
      !! This layer is for inputting 2-d data to the network.
      !! Currently, this layer must be followed by a conv2d layer.
      !! An input layer must be the first layer in the network.
      !!
      !! This is a specific function that is available
      !! under a generic name `input`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: input, layer
      !! type(layer) :: input_layer
      !! input_layer = input(28, 28)
      !! ```
      integer, intent(in) :: dim1, dim2
        !! First and second dimension sizes
      type(layer) :: res
        !! Resulting layer instance
    end function input2d

    module function input3d(dim1, dim2, dim3) result(res)
      !! 3-d input layer constructor.
      !!
      !! This is a specific function that is available
      !! under a generic name `input`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: input, layer
      !! type(layer) :: input_layer
      !! input_layer = input(28, 28, 1)
      !! ```
      integer, intent(in) :: dim1, dim2, dim3
        !! First, second and third dimension sizes
      type(layer) :: res
        !! Resulting layer instance
    end function input3d

  end interface input


  interface conv

    module function conv1d(filters, kernel_width, activation) result(res)
      !! 1-d convolutional layer constructor.
      !!
      !! This layer is for building 1-d convolutional network.
      !! Although the established convention is to call these layers 1-d,
      !! the shape of the data is actually 2-d: image width and the number of channels. 
      !! A conv1d layer must not be the first layer in the network.
      !!
      !! This specific function is available under a generic name `conv`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: conv, layer
      !! type(layer) :: conv1d_layer
      !! conv1d_layer = conv(filters=32, kernel_size=3)
      !! ```
      integer, intent(in) :: filters
        !! Number of filters in the output of the layer
      integer, intent(in) :: kernel_width
        !! Width of the convolution window, commonly 3 or 5
      class(activation_function), intent(in), optional :: activation
        !! Activation function (default sigmoid)
      type(layer) :: res
        !! Resulting layer instance
    end function conv1d

    module function conv2d(filters, kernel_width, kernel_height, activation) result(res)
      !! 2-d convolutional layer constructor.
      !!
      !! This layer is for building 2-d convolutional network.
      !! Although the established convention is to call these layers 2-d,
      !! the shape of the data is actually 3-d: image width, image height,
      !! and the number of channels.  
      !! A conv2d layer must not be the first layer in the network.
      !!
      !! This specific function is available under a generic name `conv`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: conv, layer
      !! type(layer) :: conv2d_layer  
      !! conv2d_layer = conv2d(filters=32, kernel_width=3, kernel_height=3)
      !! ```
      integer, intent(in) :: filters
        !! Number of filters in the output of the layer
      integer, intent(in) :: kernel_width
        !! Width of the convolution window, commonly 3 or 5
      integer, intent(in) :: kernel_height
        !! Height of the convolution window, commonly 3 or 5
      class(activation_function), intent(in), optional :: activation
        !! Activation function (default sigmoid)
      type(layer) :: res
        !! Resulting layer instance
    end function conv2d
    
  end interface conv


  interface maxpool

    module function maxpool1d(pool_width, stride) result(res)
      !! 1-d maxpooling layer constructor.
      !!
      !! This layer is for downscaling other layers, typically `conv1d`.
      !!
      !! This specific function is available under a generic name `maxpool`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: maxpool1d, layer
      !! type(layer) :: maxpool1d_layer
      !! maxpool1d_layer = maxpool1d(pool_width=2, stride=2)
      !! ```
      integer, intent(in) :: pool_width
        !! Width of the pooling window, commonly 2
      integer, intent(in) :: stride
        !! Stride of the pooling window, commonly equal to `pool_width`;
      type(layer) :: res
        !! Resulting layer instance
    end function maxpool1d

    module function maxpool2d(pool_width, pool_height, stride) result(res)
      !! 2-d maxpooling layer constructor.
      !!
      !! This layer is for downscaling other layers, typically `conv2d`.
      !!
      !! This specific function is available under a generic name `maxpool`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: maxpool2d, layer
      !! type(layer) :: maxpool2d_layer
      !! maxpool2d_layer = maxpool2d(pool_width=2, pool_height=2, stride=2)
      !! ```
      integer, intent(in) :: pool_width
        !! Width of the pooling window, commonly 2
      integer, intent(in) :: pool_height
        !! Height of the pooling window; currently must be equal to pool_width
      integer, intent(in) :: stride
        !! Stride of the pooling window, commonly equal to `pool_width`;
      type(layer) :: res
        !! Resulting layer instance
    end function maxpool2d

  end interface maxpool
  

  interface reshape

    module function reshape2d(dim1, dim2) result(res)
      !! Rank-1 to rank-2 reshape layer constructor.
      integer, intent(in) :: dim1, dim2
        !! Shape of the output
      type(layer) :: res
        !! Resulting layer instance
    end function reshape2d

    module function reshape3d(dim1, dim2, dim3) result(res)
      !! Rank-1 to rank-3 reshape layer constructor.
      integer, intent(in) :: dim1, dim2, dim3
        !! Shape of the output
      type(layer) :: res
        !! Resulting layer instance
    end function reshape3d

  end interface reshape


  interface

    module function dense(layer_size, activation) result(res)
      !! Dense (fully-connected) layer constructor.
      !!
      !! This layer is a building block for dense, fully-connected networks,
      !! or for an output layer of a convolutional network.
      !! A dense layer must not be the first layer in the network.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: dense, layer, relu
      !! type(layer) :: dense_layer
      !! dense_layer = dense(10)
      !! dense_layer = dense(10, activation=relu())
      !! ```
      integer, intent(in) :: layer_size
        !! The number of neurons in a dense layer
      class(activation_function), intent(in), optional :: activation
        !! Activation function instance (default sigmoid)
      type(layer) :: res
        !! Resulting layer instance
    end function dense

    module function dropout(rate) result(res)
      !! Create a dropout layer with a given dropout rate.
      !!
      !! This layer is for randomly disabling neurons during training.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: dropout, layer
      !! type(layer) :: dropout_layer
      !! dropout_layer = dropout(rate=0.5)
      !! ```
      real, intent(in) :: rate
        !! Dropout rate - fraction of neurons to randomly disable during training
      type(layer) :: res
        !! Resulting layer instance
    end function dropout

    module function flatten() result(res)
      !! Flatten (3-d -> 1-d) layer constructor.
      !!
      !! Use this layer to chain layers with 3-d outputs to layers with 1-d
      !! inputs. For example, to chain a `conv2d` or a `maxpool2d` layer
      !! with a `dense` layer for a CNN for classification, place a `flatten`
      !! layer between them.
      !!
      !! A flatten layer must not be the first layer in the network.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: flatten, layer
      !! type(layer) :: flatten_layer
      !! flatten_layer = flatten()
      !! ```
      type(layer) :: res
        !! Resulting layer instance
    end function flatten

    module function locally_connected1d(filters, kernel_size, activation) result(res)
      !! 1-d locally connected network constructor
      !!
      !! This layer is for building 1-d locally connected network.
      !! Although the established convention is to call these layers 1-d,
      !! the shape of the data is actually 2-d: image width,
      !! and the number of channels.
      !! A locally connected 1d layer must not be the first layer in the network.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: locally_connected1d, layer
      !! type(layer) :: locally_connected1d_layer
      !! locally_connected1d_layer = dense(filters=32, kernel_size=3)
      !! locally_connected1d_layer = dense(filters=32, kernel_size=3, activation='relu')
      !! ```
      integer, intent(in) :: filters
        !! Number of filters in the output of the layer
      integer, intent(in) :: kernel_size
        !! Width of the convolution window, commonly 3 or 5
      class(activation_function), intent(in), optional :: activation
        !! Activation function (default sigmoid)
      type(layer) :: res
        !! Resulting layer instance
    end function locally_connected1d

    module function linear2d(out_features) result(res)
      !! Rank-2 (sequence_length, out_features) linear layer constructor.
      !! sequence_length is determined at layer initialization, based on the
      !! output shape of the previous layer.
      integer, intent(in) :: out_features
        !! Number of output features
      type(layer) :: res
        !! Resulting layer instance
    end function linear2d

    module function self_attention(num_heads) result(res)
      !! Rank-2 (sequence_length, out_features) self attention constructor.
      !! sequence_length and model_dimension are determined at layer initialization, based on the
      !! output shape of the previous layer.
      integer, intent(in) :: num_heads
        !! Number of attention heads
      type(layer) :: res
        !! Resulting layer instance
    end function self_attention

    module function embedding(sequence_length, vocab_size, model_dimension, positional) result(res)
      !! Embedding layer constructor.
      !!
      !! This layer is for inputting token indices from the dictionary to the network.
      !! Works as a trainable lookup table that converts each index into a vector.
      !! Embedding layer must be the first layer in a network.
      integer, intent(in) :: sequence_length
        !! max len of input sequence  
      integer, intent(in) :: vocab_size
        !! length of token vocabulary
      integer, intent(in) :: model_dimension
        !! size of target embeddings
      integer, optional, intent(in) :: positional
        !! positional encoding
      type(layer) :: res
    end function embedding

    module function layernorm() result(res)
      !! Layer Normalization
      !! ((x − mean(x)) / sqrt(variance(x) + eps) * gamma + beta
      !! Based upon `Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton(2016)`:
      !! https://arxiv.org/abs/1607.06450v1
      type(layer) :: res
    end function layernorm

  end interface

end module nf_layer_constructors
