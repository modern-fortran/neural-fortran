submodule(nf_input4d_layer) nf_input4d_layer_submodule
  implicit none
contains

  pure module function input4d_layer_cons(output_shape) result(res)
    integer, intent(in) :: output_shape(4)
    type(input4d_layer) :: res
    allocate(res % output(output_shape(1), output_shape(2), &
                          output_shape(3), output_shape(4)))
    res % output = 0
  end function input4d_layer_cons

  module subroutine init(self, input_shape)
    class(input4d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
  end subroutine init

  pure module subroutine set(self, values)
    class(input4d_layer), intent(in out) :: self
    real, intent(in) :: values(:,:,:,:)
    self % output = values
  end subroutine set

end submodule nf_input4d_layer_submodule
