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

    call res % init([1])
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
end module nf_linear2d_layer
