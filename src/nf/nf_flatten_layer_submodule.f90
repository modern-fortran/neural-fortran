submodule(nf_flatten_layer) nf_flatten_layer_submodule

  !! This module provides the concrete flatten layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

contains

  elemental module function flatten_layer_cons() result(res)
    type(flatten_layer) :: res
  end function flatten_layer_cons


  pure module subroutine backward(self, input, gradient)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:)
    self % gradient = reshape(gradient, shape(input))
  end subroutine backward


  pure module subroutine forward(self, input)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    self % output = pack(input, .true.)
  end subroutine forward


  module subroutine init(self, input_shape)
    class(flatten_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_shape = input_shape
    self % output_size = product(input_shape)

    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3)))
    self % gradient = 0

    allocate(self % output(self % output_size))
    self % output = 0

  end subroutine init

end submodule nf_flatten_layer_submodule
