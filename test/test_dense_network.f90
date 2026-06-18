program test_dense_network
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, input, network, sgd
  use tuff, only: test, test_result
  implicit none
  type(network) :: net
  type(test_result) :: tests

  ! Minimal 2-layer network
  net = network([ &
    input(1), &
    dense(1) &
  ])

  tests = test("test_dense_network", [ &
    test("network has 2 layers", size(net % layers) == 2), &
    test("network predicts 0.5 for input 0", all(net % predict([0.]) == 0.5)), &
    test(simple_training), &
    test(larger_network_size) &
  ])

contains

  type(test_result) function simple_training() result(res)
    real :: x(1), y(1)
    real :: tolerance = 1e-3
    integer :: n
    integer, parameter :: num_iterations = 1000
    type(network) :: net 

    res % name = 'simple training'

    ! Minimal 2-layer network
    net = network([ &
      input(1), &
      dense(1) &
    ])

    x = [0.123]
    y = [0.765]

    do n = 1, num_iterations
      call net % forward(x)
      call net % backward(y)
      call net % update(sgd(learning_rate=1.))
      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do

    res % ok = n <= num_iterations

  end function simple_training

  type(test_result) function larger_network_size() result(res)
    type(network) :: net 

    res % name = 'larger network training'

    ! A bit larger multi-layer network
    net = network([ &
      input(784), &
      dense(30), &
      dense(20), &
      dense(10) &
    ])

    res % ok = size(net % layers) == 4

  end function larger_network_size

end program test_dense_network