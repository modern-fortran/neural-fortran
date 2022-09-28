submodule(nf_reshape_layer) nf_reshape_layer_submodule

  use nf_base_layer, only: base_layer

  implicit none

contains

  pure module function reshape3d_layer_cons(output_shape) result(res)
    integer, intent(in) :: output_shape(3)
    type(reshape3d_layer) :: res
    res % output_shape = output_shape
  end function reshape3d_layer_cons


  pure module subroutine backward(self, input, gradient)
    class(reshape3d_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:,:,:)
    ! The `input` dummy argument is not used but nevertheless declared
    ! because the abstract type requires it.
    self % gradient = pack(gradient, .true.)
  end subroutine backward


  pure module subroutine forward(self, input)
    class(reshape3d_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    self % output = reshape(input, self % output_shape)
  end subroutine forward


  module subroutine init(self, input_shape)
    class(reshape3d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_shape = input_shape

    allocate(self % gradient(input_shape(1)))
    self % gradient = 0

    allocate(self % output( &
      self % output_shape(1), &
      self % output_shape(2), &
      self % output_shape(3) &
      ))
    self % output = 0

  end subroutine init

end submodule nf_reshape_layer_submodule
