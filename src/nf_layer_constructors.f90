module nf_layer_constructors

  !! This module provides the functions to instantiate specific layers.

  use nf_layer, only: layer

  implicit none

  private
  public :: conv2d, dense, input

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
      !! use nf, only :: dense, layer
      !! type(layer) :: dense_layer
      !! dense_layer = dense(10)
      !! dense_layer = dense(10, activation='relu')
      !! ```
      integer, intent(in) :: layer_size
        !! The number of neurons in a dense layer
      character(*), intent(in), optional :: activation
        !! Activation function (default 'sigmoid')
      type(layer) :: res
        !! Resulting layer instance
    end function dense

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
      character(*), intent(in), optional :: activation
        !! Activation function (default 'sigmoid')
      type(layer) :: res
        !! Resulting layer instance
    end function conv2d

  end interface

end module nf_layer_constructors
