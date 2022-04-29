module nf_conv2d_layer

  !! This is a placeholder module that will later define a concrete conv2d
  !! layer type.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: conv2d_layer

  type, extends(base_layer) :: conv2d_layer

    integer :: width
    integer :: height
    integer :: channels
    integer :: window_size
    integer :: filters

    real, allocatable :: biases(:) ! as many as there are filters
    real, allocatable :: kernel(:,:,:,:)
    real, allocatable :: output(:,:,:)

  contains

    procedure :: init
    procedure :: forward
    procedure :: backward

  end type conv2d_layer

  interface conv2d_layer
    module procedure :: conv2d_layer_cons
  end interface conv2d_layer

contains

  pure function conv2d_layer_cons(window_size, filters, activation) result(res)
    integer, intent(in) :: window_size
    integer, intent(in) :: filters
    character(*), intent(in) :: activation
    type(conv2d_layer) :: res
    res % window_size = window_size
    res % filters = filters
    call res % set_activation(activation)
  end function conv2d_layer_cons


  subroutine init(self, input_shape)
    class(conv2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % width = input_shape(1) - self % window_size + 1
    self % height = input_shape(2) - self % window_size + 1
    self % channels = input_shape(3)

    allocate(self % output(self % width, self % height, self % filters))
    self % output = 0

    allocate(self % kernel(self % window_size, self % window_size, &
                           self % channels, self % filters))
    self % kernel = 0 ! TODO 4-d randn

    allocate(self % biases(self % filters))
    self % biases = 0

  end subroutine init


  subroutine forward(self, input)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    print *, 'Warning: conv2d forward pass not implemented'
  end subroutine forward


  subroutine backward(self, input, gradient)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    print *, 'Warning: conv2d backward pass not implemented'
  end subroutine backward

end module nf_conv2d_layer
