program test_linear2d_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  logical :: ok = .true.
  real :: sample_input(2, 3, 4) = reshape(&
      [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2,&
       0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2],&
      [2, 3, 4]) ! first batch are 0.1, second 0.2
  type(linear2d_layer) :: linear

  linear = linear2d_layer(batch_size=2, sequence_length=3, in_features=4, out_features=1)

  call test_linear2d_layer_forward(linear, ok, sample_input)

contains
  subroutine test_linear2d_layer_forward(linear, ok, input)
    type(linear2d_layer), intent(in out) :: linear
    logical, intent(in out) :: ok
    real, intent(in) :: input(2, 3, 4)
    real :: output_shape(3)
    real :: output_flat(6)
    real :: expected_shape(3) = [2, 3, 1]
    real :: expected_output_flat(6) = [0.15, 0.19, 0.15, 0.19, 0.15, 0.19]

    call linear % forward(input)

    output_shape = shape(linear % output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect shape.. failed'
    end if
    output_flat = reshape(linear % output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if
  end subroutine test_linear2d_layer_forward
end program test_linear2d_layer