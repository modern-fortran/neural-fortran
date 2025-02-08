module nf_locally_connected_1d_layer
  !! This module provides a locally connected 1d layer type.

  use nf_activation, only: activation_function
  use nf_base_layer,   only: base_layer
  implicit none

  private
  public :: locally_connected_1d_layer

  type, extends(base_layer) :: locally_connected_1d_layer
    ! For a 1D layer, we assume an input shape of [channels, input_length]
    integer :: channels        ! number of input channels
    integer :: input_length    ! length of the 1D input
    integer :: output_length   ! computed as input_length - kernel_size + 1
    integer :: kernel_size     ! size of the 1D window
    integer :: filters         ! number of filters (output channels)

    ! Parameters (unshared weights)
    ! Kernel shape: (filters, output_length, channels, kernel_size)
    real, allocatable :: kernel(:,:,:,:)
    ! Biases shape: (filters, output_length)
    real, allocatable :: biases(:,:)

    ! Forward-pass arrays
    ! Pre-activation values: shape (filters, output_length)
    real, allocatable :: z(:,:)
    ! Activated output: shape (filters, output_length)
    real, allocatable :: output(:,:)

    ! Gradients for backpropagation
    ! Gradient for kernel, same shape as kernel
    real, allocatable :: dw(:,:,:,:)
    ! Gradient for biases, same shape as biases
    real, allocatable :: db(:,:)
    ! Gradient with respect to the input, shape (channels, input_length)
    real, allocatable :: gradient(:,:)

    ! Activation function
    class(activation_function), allocatable :: activation
  contains
    procedure :: forward
    procedure :: backward
    procedure :: get_gradients
    procedure :: get_num_params
    procedure :: get_params
    procedure :: init
    procedure :: set_params
  end type locally_connected_1d_layer

  interface locally_connected_1d_layer
    module function locally_connected_1d_layer_cons(filters, kernel_size, activation) result(res)
      !! Constructor for the locally connected 1d layer.
      integer, intent(in)                   :: filters
      integer, intent(in)                   :: kernel_size
      class(activation_function), intent(in):: activation
      type(locally_connected_1d_layer)       :: res
    end function locally_connected_1d_layer_cons
  end interface locally_connected_1d_layer

  interface
    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !! input_shape: integer array of length 2, where
      !!   input_shape(1) = number of channels
      !!   input_shape(2) = input length
      class(locally_connected_1d_layer), intent(inout) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module subroutine forward(self, input)
      !! Apply the forward pass.
      !! Input shape: (channels, input_length)
      class(locally_connected_1d_layer), intent(inout) :: self
      real, intent(in) :: input(:,:)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward pass.
      !! input: shape (channels, input_length)
      !! gradient: gradient w.r.t. output, shape (filters, output_length)
      class(locally_connected_1d_layer), intent(inout) :: self
      real, intent(in) :: input(:,:)
      real, intent(in) :: gradient(:,:)
    end subroutine backward

    pure module function get_num_params(self) result(num_params)
      !! Get the total number of parameters (kernel + biases)
      class(locally_connected_1d_layer), intent(in) :: self
      integer :: num_params
    end function get_num_params

    module function get_params(self) result(params)
      !! Return a flattened vector of parameters (kernel then biases).
      class(locally_connected_1d_layer), intent(in), target :: self
      real, allocatable :: params(:)
    end function get_params

    module function get_gradients(self) result(gradients)
      !! Return a flattened vector of gradients (dw then db).
      class(locally_connected_1d_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients

    module subroutine set_params(self, params)
      !! Set the parameters from a flattened vector.
      class(locally_connected_1d_layer), intent(inout) :: self
      real, intent(in) :: params(:)
    end subroutine set_params
  end interface

end module nf_locally_connected_1d_layer
