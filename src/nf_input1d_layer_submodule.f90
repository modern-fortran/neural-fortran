submodule(nf_input1d_layer) nf_input1d_layer_submodule
  implicit none
contains

  pure module function input1d_layer_cons(output_size) result(res)
    integer, intent(in) :: output_size
    type(input1d_layer) :: res
    allocate(res % output(output_size))
    res % output = 0
  end function input1d_layer_cons

  module subroutine init(self, input_shape)
    class(input1d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
  end subroutine init

  pure module subroutine set(self, values)
    class(input1d_layer), intent(in out) :: self
    real, intent(in) :: values(:)
    self % output = values
  end subroutine set

end submodule nf_input1d_layer_submodule
