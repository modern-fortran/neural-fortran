program test_dense_network
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, input, network
  implicit none
  type(network) :: net
  logical :: ok = .true.

  ! Minimal 2-layer network
  net = network([ &
    input(1), &
    dense(1) &
  ])

  if (.not. size(net % layers) == 2) then
    write(stderr, '(a)') 'dense network should have 2 layers.. failed'
    ok = .false.
  end if

  if (.not. all(net % predict([0.]) == 0.5)) then
    write(stderr, '(a)') &
      'dense network should output exactly 0.5 for input 0.. failed'
    ok = .false.
  end if

  training: block
    real :: x(1), y(1)
    real :: tolerance = 1e-3
    integer :: n
    integer, parameter :: num_iterations = 1000 

    x = [0.123]
    y = [0.765]

    do n = 1, num_iterations
      call net % forward(x)
      call net % backward(y)
      call net % update(1.)
      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'dense network should converge in simple training.. failed'
      ok = .false.
    end if

  end block training

  ! A bit larger multi-layer network
  net = network([ &
    input(784), &
    dense(30), &
    dense(20), &
    dense(10) &
  ])

  if (.not. size(net % layers) == 4) then
    write(stderr, '(a)') 'dense network should have 4 layers.. failed'
    ok = .false.
  end if

  if (ok) then
    print '(a)', 'test_dense_network: All tests passed.'
  else
    write(stderr, '(a)') 'test_dense_network: One or more tests failed.'
    stop 1
  end if

end program test_dense_network
