module nf_linear2d_layer

  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: linear2d_layer

  type, extends(base_layer) :: linear2d_layer
    integer :: batch_size, sequence_length, in_features, out_features

    real, allocatable :: weights(:, :)
    real, allocatable :: biases(:)
    real, allocatable :: output(:, :, :)
    real, allocatable :: gradient(:, :, :) ! input gradient
    real, allocatable :: dw(:, :) ! weight gradients
    real, allocatable :: db(:) ! bias gradients

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params

  end type linear2d_layer

  interface linear2d_layer
    module function linear2d_layer_cons(in_features, out_features) &
      result(res)
      integer, intent(in) :: in_features, out_features
      type(linear2d_layer) :: res
    end function linear2d_layer_cons
  end interface linear2d_layer

  interface
    pure module subroutine forward(self, input)
      class(linear2d_layer), intent(in out) :: self
      real, intent(in) :: input(:, :, :)
    end subroutine forward

    module subroutine init(self, input_shape)
      class(linear2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init
  end interface

contains
  module function linear2d_layer_cons(&
      batch_size, sequence_length, in_features, out_features&
  ) result(res)
    integer, intent(in) :: batch_size, sequence_length, in_features, out_features
    type(linear2d_layer) :: res

    res % in_features = in_features
    res % out_features = out_features
    res % sequence_length = sequence_length
    res % batch_size = batch_size
  end function linear2d_layer_cons

  module subroutine init(self, input_shape)
    class(linear2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    allocate(self % output(self % batch_size, self % sequence_length, self % out_features))
    allocate(self % gradient(self % batch_size, self % sequence_length, self % in_features))

    allocate(self % weights(self % in_features, self % out_features))
    self % weights = 0.1

    allocate(self % biases(self % out_features))
    self%biases = 0.11

    allocate(self % dw(self % in_features, self % out_features))
    self % dw = 0.0
    allocate(self % db(self % out_features))
    self % db = 0.0
  end subroutine init

  pure module subroutine forward(self, input)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :, :)
    integer :: i, j

    do concurrent(i = 1: self % batch_size)
      self % output(i, :, :) = matmul(input(i, :, :), self % weights)
    end do
    do concurrent(i = 1: self % batch_size, j = 1: self % sequence_length)
      self % output(i, j, :) = self % output(i, j, :) + self % biases
    end do
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :, :)
    real, intent(in) :: gradient(:, :, :)
    real :: db(self % out_features)
    real :: dw(self % in_features, self % out_features)
    integer :: i

    do concurrent(i = 1: self % batch_size)
      self % dw = self % dw + matmul(transpose(input(i, :, :)), gradient(i, :, :))
      self % db = self % db + sum(gradient(i, :, :), 1)
      self % gradient(i, :, :) = matmul(gradient(i, :, :), transpose(self % weights))
    end do
  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(linear2d_layer), intent(in) :: self
    integer :: num_params

    ! Number of weigths times number of biases
    num_params = self % in_features * self % out_features + self % out_features

  end function get_num_params


  module function get_params(self) result(params)
    class(linear2d_layer), intent(in), target :: self
    real, allocatable :: params(:)

    real, pointer :: w_(:) => null()

    w_(1:size(self % weights)) => self % weights

    params = [ &
      w_, &
      self % biases &
    ]

  end function get_params


  module function get_gradients(self) result(gradients)
    class(linear2d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)

    real, pointer :: dw_(:) => null()

    dw_(1:size(self % dw)) => self % dw

    gradients = [ &
      dw_, &
      self % db &
    ]

  end function get_gradients


  module subroutine set_params(self, params)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in), target :: params(:)

    real, pointer :: p_(:,:) => null()

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    associate(n => self % in_features * self % out_features)
      ! reshape the weights
      p_(1:self % in_features, 1:self % out_features) => params(1 : n)
      self % weights = p_

      ! reshape the biases
      self % biases = params(n + 1 : n + self % out_features)
    end associate

  end subroutine set_params
end module nf_linear2d_layer
