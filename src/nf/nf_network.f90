module nf_network

  !! This module provides the network type to create new models.

  use nf_layer, only: layer
  use nf_metrics, only: metric_type
  use nf_loss, only: loss_type
  use nf_optimizers, only: optimizer_base_type

  implicit none

  private
  public :: network

  type :: network

    type(layer), allocatable :: layers(:)
    class(loss_type), allocatable :: loss

  contains

    procedure :: backward
    procedure :: get_num_params
    procedure :: get_params
    procedure :: print_info
    procedure :: set_params
    procedure :: set_training_mode
    procedure :: train
    procedure :: update

    procedure, private :: evaluate_batch_1d
    procedure, private :: forward_1d
    procedure, private :: forward_1d_int
    procedure, private :: forward_2d
    procedure, private :: forward_3d
    procedure, private :: get_output_1d
    procedure, private :: predict_1d
    procedure, private :: predict_1d_int
    procedure, private :: predict_2d
    procedure, private :: predict_3d
    procedure, private :: predict_batch_1d
    procedure, private :: predict_batch_3d

    generic :: evaluate => evaluate_batch_1d
    generic :: forward => forward_1d, forward_1d_int, forward_2d, forward_3d
    generic :: get_output => get_output_1d
    generic :: predict => predict_1d, predict_1d_int, predict_2d, predict_3d
    generic :: predict_batch => predict_batch_1d, predict_batch_3d

  end type network

  interface network

    module function network_from_layers(layers) result(res)
      !! Create a new `network` instance from an array of `layer` instances.
      type(layer), intent(in) :: layers(:)
        !! Input array of `layer` instances;
        !! the first element must be an input layer.
      type(network) :: res
        !! An instance of the `network` type
    end function network_from_layers

    module function network_from_keras(filename) result(res)
      !! Create a new `network` instance
      !! from a Keras model saved in an h5 file.
      character(*), intent(in) :: filename
        !! Path to the Keras model h5 file
      type(network) :: res
        !! An instance of the `network` type
    end function network_from_keras

  end interface network

  interface evaluate
    module function evaluate_batch_1d(self, input_data, output_data, metric) result(res)
      class(network), intent(in out) :: self
      real, intent(in) :: input_data(:,:)
      real, intent(in) :: output_data(:,:)
      class(metric_type), intent(in), optional :: metric
      real, allocatable :: res(:,:)
    end function evaluate_batch_1d
  end interface evaluate

  interface forward

    module subroutine forward_1d(self, input)
      !! Apply a forward pass through the network.
      !!
      !! This changes the state of layers on the network.
      !! Typically used only internally from the `train` method,
      !! but can be invoked by the user when creating custom optimizers.
      !!
      !! This specific subroutine is for 1-d input data.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:)
        !! 1-d input data
    end subroutine forward_1d

    module subroutine forward_1d_int(self, input)
      !! Same as `forward_1d` except `integer`
      class(network), intent(in out) :: self
      integer, intent(in) :: input(:)
    end subroutine forward_1d_int

    module subroutine forward_2d(self, input)
      !! Apply a forward pass through the network.
      !!
      !! This changes the state of layers on the network.
      !! Typically used only internally from the `train` method,
      !! but can be invoked by the user when creating custom optimizers.
      !!
      !! This specific subroutine is for 1-d input data.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:)
        !! 2-d input data
    end subroutine forward_2d

    module subroutine forward_3d(self, input)
      !! Apply a forward pass through the network.
      !!
      !! This changes the state of layers on the network.
      !! Typically used only internally from the `train` method,
      !! but can be invoked by the user when creating custom optimizers.
      !!
      !! This specific subroutine is for 3-d input data.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:,:)
        !! 3-d input data
    end subroutine forward_3d

  end interface forward

  interface predict

    module function predict_1d(self, input) result(res)
      !! Return the output of the network given the input 1-d array.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:)
        !! Input data
      real, allocatable :: res(:)
        !! Output of the network
    end function predict_1d

    module function predict_1d_int(self, input) result(res)
      !! Same as `predict_1d` except `integer`
      class(network), intent(in out) :: self
      integer, intent(in) :: input(:)
      real, allocatable :: res(:)
    end function predict_1d_int

    module function predict_2d(self, input) result(res)
      !! Return the output of the network given the input 1-d array.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:)
        !! Input data
      real, allocatable :: res(:)
        !! Output of the network
    end function predict_2d

    module function predict_3d(self, input) result(res)
      !! Return the output of the network given the input 3-d array.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:,:)
        !! Input data
      real, allocatable :: res(:)
        !! Output of the network
    end function predict_3d

  end interface predict

  interface predict_batch
    module function predict_batch_1d(self, input) result(res)
      !! Return the output of the network given an input batch of 3-d data.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:)
        !! Input data; the last dimension is the batch
      real, allocatable :: res(:,:)
        !! Output of the network; the last dimension is the batch
    end function predict_batch_1d

    module function predict_batch_3d(self, input) result(res)
      !! Return the output of the network given an input batch of 3-d data.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input(:,:,:,:)
        !! Input data; the last dimension is the batch
      real, allocatable :: res(:,:)
        !! Output of the network; the last dimension is the batch
    end function predict_batch_3d
  end interface predict_batch

  interface get_output
    module subroutine get_output_1d(self, output)
      class(network), intent(in), target :: self
      real, pointer, intent(out) :: output(:)
    end subroutine get_output_1d
  end interface get_output

  interface

    module subroutine backward(self, output, loss, gradient)
      !! Apply one backward pass through the network.
      !! This changes the state of layers on the network.
      !! Typically used only internally from the `train` method,
      !! but can be invoked by the user when creating custom optimizers.
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: output(:)
        !! Output data
      class(loss_type), intent(in), optional :: loss
        !! Loss instance to use. If not provided, the default is quadratic().
      real, intent(in), optional :: gradient(:)
        !! Gradient to use for the output layer.
        !! If not provided, the gradient in the last layer is computed using
        !! the loss function.
        !! Passing the gradient is useful for merging/concatenating multiple
        !! networks.
    end subroutine backward

    module integer function get_num_params(self)
      !! Get the number of parameters in the network.
      class(network), intent(in) :: self
      !! Network instance
    end function get_num_params

    module function get_params(self) result(params)
      !! Get the network parameters (weights and biases).
      class(network), intent(in) :: self
        !! Network instance
      real, allocatable :: params(:)
        !! Network parameters to get
    end function get_params

    module subroutine set_params(self, params)
      !! Set the network parameters (weights and biases).
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: params(:)
        !! Network parameters to set
    end subroutine set_params

    module subroutine set_training_mode(self, training)
      !! Set the mode to training (.true.) or inference (.false.).
      !! Used internally to enable/disable the dropout layers in the network.
      class(network), intent(in out) :: self
        !! Network instance
      logical, intent(in) :: training
        !! .true. for training mode, .false. for inference.
    end subroutine set_training_mode

    module subroutine print_info(self)
      !! Prints a brief summary of the network and its layers to the screen.
      class(network), intent(in) :: self
        !! Network instance
    end subroutine print_info

    module subroutine train(self, input_data, output_data, batch_size, &
                            epochs, optimizer, loss)
      class(network), intent(in out) :: self
        !! Network instance
      real, intent(in) :: input_data(:,:)
        !! Input data to train on;
        !! first dimension contains a single sample
        !! and its size must match the size of the input layer.
      real, intent(in) :: output_data(:,:)
        !! Output data to train on;
        !! first dimension contains a single sample
        !! and its size must match the size of the input layer.
      integer, intent(in) :: batch_size
        !! Batch size to use.
        !! Set to 1 for a pure stochastic gradient descent.
        !! Set to `size(input_data, dim=2)` for a batch gradient descent.
      integer, intent(in) :: epochs
        !! Number of epochs to run
      class(optimizer_base_type), intent(in), optional :: optimizer
        !! Optimizer instance to use. If not provided, the default is sgd().
      class(loss_type), intent(in), optional :: loss
        !! Loss instance to use. If not provided, the default is quadratic().
    end subroutine train

    module subroutine update(self, optimizer, batch_size)
      !! Update the weights and biases on all layers using the stored
      !! gradients (from backward passes) on those layers, and flush those
      !! same stored gradients to zero.
      !! This changes the state of layers on the network.
      !! Typically used only internally from the `train` method,
      !! but can be invoked by the user when creating custom optimizers.
      class(network), intent(in out) :: self
        !! Network instance
      class(optimizer_base_type), intent(in), optional :: optimizer
        !! Optimizer instance to use
      integer, intent(in), optional :: batch_size
        !! Batch size to use.
        !! Set to 1 for a pure stochastic gradient descent (default).
        !! Set to `size(input_data, dim=2)` for a batch gradient descent.
    end subroutine update

  end interface

end module nf_network
