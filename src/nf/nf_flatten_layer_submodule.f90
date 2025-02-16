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


  pure module subroutine backward_2d(self, input, gradient)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:)
    self % gradient_2d = reshape(gradient, shape(input))
  end subroutine backward_2d


  pure module subroutine backward_3d(self, input, gradient)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:)
    self % gradient_3d = reshape(gradient, shape(input))
  end subroutine backward_3d


  pure module subroutine forward_2d(self, input)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    self % output = pack(input, .true.)
  end subroutine forward_2d


  pure module subroutine forward_3d(self, input)
    class(flatten_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    self % output = pack(input, .true.)
  end subroutine forward_3d


  module subroutine init(self, input_shape)
    class(flatten_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_shape = input_shape
    self % output_size = product(input_shape)

    if (size(input_shape) == 2) then
      allocate(self % gradient_2d(input_shape(1), input_shape(2)))
      self % gradient_2d = 0
    else if (size(input_shape) == 3) then
      allocate(self % gradient_3d(input_shape(1), input_shape(2), input_shape(3)))
      self % gradient_3d = 0
    end if

    allocate(self % output(self % output_size))
    self % output = 0

  end subroutine init

end submodule nf_flatten_layer_submodule
