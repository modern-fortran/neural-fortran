program test_conv2d_network

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, input, network, dense, sgd, maxpool2d

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:,:,:), output(:,:,:)
  logical :: ok = .true.

  ! 3-layer convolutional network
  net = network([ &
    input([3, 32, 32]), &
    conv2d(filters=16, kernel_size=3), &
    conv2d(filters=32, kernel_size=3) &
  ])

  if (.not. size(net % layers) == 3) then
    write(stderr, '(a)') 'conv2d network should have 3 layers.. failed'
    ok = .false.
  end if

  ! Test for output shape
  allocate(sample_input(3, 32, 32))
  sample_input = 0

  call net % forward(sample_input)
  call net % layers(3) % get_output(output)

  if (.not. all(shape(output) == [32, 28, 28])) then
    write(stderr, '(a)') 'conv2d network output should have correct shape.. failed'
    ok = .false.
  end if

  deallocate(sample_input, output)

  ! Test training of a minimal constant mapping
  allocate(sample_input(1, 5, 5))
  call random_number(sample_input)

  net = network([ &
    input(shape(sample_input)), &
    conv2d(filters=1, kernel_size=3), &
    conv2d(filters=1, kernel_size=3), &
    dense(1) &
  ])

  training: block
    real :: y(1)
    real :: tolerance = 1e-5
    integer :: n
    integer, parameter :: num_iterations = 1000

    y = [0.1234567]

    do n = 1, num_iterations
      call net % forward(sample_input)
      call net % backward(y)
      call net % update(optimizer=sgd(learning_rate=1.))
      if (all(abs(net % predict(sample_input) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'convolutional network should converge in simple training.. failed'
      ok = .false.
    end if

  end block training

  if (ok) then
    print '(a)', 'test_conv2d_network: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_network: One or more tests failed.'
    stop 1
  end if

end program test_conv2d_network
