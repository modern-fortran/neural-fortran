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
  end subroutine backward


  pure module subroutine forward(self, input)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
  end subroutine forward


  module subroutine init(self, input_shape)
    class(flatten_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
  end subroutine init

end submodule nf_flatten_layer_submodule
