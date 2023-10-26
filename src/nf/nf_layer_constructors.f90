module nf_layer_constructors

  !! This module provides the functions to instantiate specific layers.

  use nf_layer, only: layer
  use nf_activation, only : activation_function

  implicit none

  private
  public :: conv2d, dense, flatten, input, maxpool2d, reshape, rnn

  interface input

    pure module function input1d(layer_size) result(res)
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

    pure module function input3d(layer_shape) result(res)
      !! 3-d input layer constructor.
      !!
      !! This layer is for inputting 3-d data to the network.
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
      !! input_layer = input([28, 28, 1])
      !! ```
      integer, intent(in) :: layer_shape(3)
        !! Shape of the input layer
      type(layer) :: res
        !! Resulting layer instance
    end function input3d

  end interface input

  interface

    pure module function dense(layer_size, activation) result(res)
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

    pure module function flatten() result(res)
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

    pure module function conv2d(filters, kernel_size, activation) result(res)
      !! 2-d convolutional layer constructor.
      !!
      !! This layer is for building 2-d convolutional network.
      !! Although the established convention is to call these layers 2-d,
      !! the shape of the data is actuall 3-d: image width, image height,
      !! and the number of channels.
      !! A conv2d layer must not be the first layer in the network.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: conv2d, layer
      !! type(layer) :: conv2d_layer
      !! conv2d_layer = dense(filters=32, kernel_size=3)
      !! conv2d_layer = dense(filters=32, kernel_size=3, activation='relu')
      !! ```
      integer, intent(in) :: filters
        !! Number of filters in the output of the layer
      integer, intent(in) :: kernel_size
        !! Width of the convolution window, commonly 3 or 5
      class(activation_function), intent(in), optional :: activation
        !! Activation function (default sigmoid)
      type(layer) :: res
        !! Resulting layer instance
    end function conv2d

    pure module function maxpool2d(pool_size, stride) result(res)
      !! 2-d maxpooling layer constructor.
      !!
      !! This layer is for downscaling other layers, typically `conv2d`.
      !!
      !! Example:
      !!
      !! ```
      !! use nf, only :: maxpool2d, layer
      !! type(layer) :: maxpool2d_layer
      !! maxpool2d_layer = maxpool2d(pool_size=2)
      !! maxpool2d_layer = maxpool2d(pool_size=2, stride=3)
      !! ```
      integer, intent(in) :: pool_size
        !! Width of the pooling window, commonly 2
      integer, intent(in), optional :: stride
        !! Stride of the pooling window, commonly equal to `pool_size`;
        !! Defaults to `pool_size` if omitted.
      type(layer) :: res
        !! Resulting layer instance
    end function maxpool2d

    pure module function reshape(output_shape) result(res)
      !! Rank-1 to rank-any reshape layer constructor.
      !! Currently implemented is only rank-3 for the output of the reshape.
      !!
      !! This layer is for connecting 1-d inputs to conv2d or similar layers.
      integer, intent(in) :: output_shape(:)
        !! Shape of the output
      type(layer) :: res
        !! Resulting layer instance
    end function reshape

  end interface

end module nf_layer_constructors
