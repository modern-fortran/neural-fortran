module nf_conv2d_layer

  !! This modules provides a 2-d convolutional `conv2d_layer` type.

  use nf_activation_3d, only: activation_function
  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: conv2d_layer

  type, extends(base_layer) :: conv2d_layer

    integer :: width
    integer :: height
    integer :: channels
    integer :: kernel_size
    integer :: filters

    real, allocatable :: biases(:) ! size(filters)
    real, allocatable :: kernel(:,:,:,:) ! filters x channels x window x window
    real, allocatable :: output(:,:,:) ! filters x output_width * output_height
    real, allocatable :: z(:,:,:) ! kernel .dot. input + bias

    real, allocatable :: dw(:,:,:,:) ! weight (kernel) gradients
    real, allocatable :: db(:) ! bias gradients
    real, allocatable :: gradient(:,:,:)

    procedure(activation_function), pointer, nopass :: &
      activation => null()
    procedure(activation_function), pointer, nopass :: &
      activation_prime => null()

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward
    procedure :: get_num_params
    procedure :: get_params
    procedure :: set_params
    procedure :: set_activation
    procedure :: update

  end type conv2d_layer

  interface conv2d_layer
    pure module function conv2d_layer_cons(filters, kernel_size, activation) &
      result(res)
      !! `conv2d_layer` constructor function
      integer, intent(in) :: filters
      integer, intent(in) :: kernel_size
      character(*), intent(in) :: activation
      type(conv2d_layer) :: res
    end function conv2d_layer_cons
  end interface conv2d_layer

  interface

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      integer, intent(in) :: input_shape(:)
        !! Input layer dimensions
    end subroutine init

    pure module subroutine forward(self, input)
      !! Apply a forward pass on the `conv2d` layer.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      !! Apply a backward pass on the `conv2d` layer.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      real, intent(in) :: input(:,:,:)
        !! Input data (previous layer)
      real, intent(in) :: gradient(:,:,:)
        !! Gradient (next layer)
    end subroutine backward

    pure module function get_num_params(self) result(num_params)
      !! Get the number of parameters in the layer.
      class(conv2d_layer), intent(in) :: self
        !! A `conv2d_layer` instance
      integer :: num_params
        !! Number of parameters
    end function get_num_params

    pure module function get_params(self) result(params)
      !! Get the parameters of the layer.
      class(conv2d_layer), intent(in) :: self
        !! A `conv2d_layer` instance
      real, allocatable :: params(:)
        !! Parameters to get
    end function get_params

    module subroutine set_params(self, params)
      !! Set the parameters of the layer.
      class(conv2d_layer), intent(in out) :: self
        !! A `conv2d_layer` instance
      real, intent(in) :: params(:)
        !! Parameters to set
    end subroutine set_params

    elemental module subroutine set_activation(self, activation)
    !! Set the activation functions.
    class(conv2d_layer), intent(in out) :: self
      !! Layer instance
    character(*), intent(in) :: activation
      !! String with the activation function name
    end subroutine set_activation

    module subroutine update(self, learning_rate)
      !! Update the weights and biases.
      class(conv2d_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: learning_rate
        !! Learning rate (must be > 0)
    end subroutine update

  end interface

end module nf_conv2d_layer
