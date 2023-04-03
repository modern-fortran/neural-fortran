program test_parametric_activation
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, layer
  use nf_dense_layer, only: dense_layer
  use nf_activation, only: elu, leaky_relu
  implicit none
  type(layer) :: layer1
  real :: alpha
  logical :: ok = .true.

  layer1 = dense(10, activation=elu())

  select type(this_layer => layer1 % p)
    type is (dense_layer)
      select type(this_activation => this_layer % activation)
        type is (elu)
          alpha = this_activation % alpha
      end select
  end select

  if (.not. alpha == 1) then
    ok = .false.
    write(stderr, '(a)') 'default alpha for ELU is as expected.. failed'
  end if

  layer1 = dense(10, activation=elu(0.1))

  select type(this_layer => layer1 % p)
    type is (dense_layer)
      select type(this_activation => this_layer % activation)
        type is (elu)
          alpha = this_activation % alpha
      end select
  end select

  if (.not. alpha == 0.1) then
    ok = .false.
    write(stderr, '(a)') 'User set alpha for ELU is as expected.. failed'
  end if

  layer1 = dense(10, activation=leaky_relu())

  select type(this_layer => layer1 % p)
    type is (dense_layer)
      select type(this_activation => this_layer % activation)
        type is (leaky_relu)
          alpha = this_activation % alpha
      end select
  end select

  if (.not. alpha == 0.3) then
    ok = .false.
    write(stderr, '(a)') 'Default alpha for leaky ReLU is as expected.. failed'
  end if

  layer1 = dense(10, activation=leaky_relu(0.01))

  select type(this_layer => layer1 % p)
    type is (dense_layer)
      select type(this_activation => this_layer % activation)
        type is (leaky_relu)
          alpha = this_activation % alpha
      end select
  end select

  if (.not. alpha == 0.01) then
    ok = .false.
    write(stderr, '(a)') 'User set alpha for leaky ReLU is as expected.. failed'
  end if

  if (ok) then
    print '(a)', 'test_parametric_activation: All tests passed.'
  else
    write(stderr, '(a)') 'test_parametric_activation: One or more tests failed.'
    stop 1
  end if

end program test_parametric_activation
