submodule(nf_input2d_layer) nf_input2d_layer_submodule
  implicit none
contains

  pure module function input2d_layer_cons(output_shape) result(res)
    integer, intent(in) :: output_shape(2)
    type(input2d_layer) :: res
    allocate(res % output(output_shape(1), output_shape(2)))
    res % output = 0
  end function input2d_layer_cons

  module subroutine init(self, input_shape)
    class(input2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
  end subroutine init

  pure module subroutine set(self, values)
    class(input2d_layer), intent(in out) :: self
    real, intent(in) :: values(:,:)
    self % output = values
  end subroutine set

end submodule nf_input2d_layer_submodule 