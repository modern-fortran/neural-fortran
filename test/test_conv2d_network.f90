program test_conv2d_network

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv2d, input, network, dense, sgd, maxpool2d

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:,:,:), output(:,:,:)
  logical :: ok = .true.

  ! 3-layer convolutional network
  net = network([ &
    input(3, 32, 32), &
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

  training1: block

    type(network) :: cnn
    real :: y(1)
    real :: tolerance = 1e-5
    integer :: n
    integer, parameter :: num_iterations = 1000

    ! Test training of a minimal constant mapping
    allocate(sample_input(1, 5, 5))
    call random_number(sample_input)

    cnn = network([ &
      input(1, 5, 5), &
      conv2d(filters=1, kernel_size=3), &
      conv2d(filters=1, kernel_size=3), &
      dense(1) &
    ])

    y = [0.1234567]

    do n = 1, num_iterations
      call cnn % forward(sample_input)
      call cnn % backward(y)
      call cnn % update(optimizer=sgd(learning_rate=1.))
      if (all(abs(cnn % predict(sample_input) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'convolutional network 1 should converge in simple training.. failed'
      ok = .false.
    end if

  end block training1

  training2: block

    type(network) :: cnn
    real :: x(1, 8, 8)
    real :: y(1)
    real :: tolerance = 1e-5
    integer :: n
    integer, parameter :: num_iterations = 1000

    call random_number(x)
    y = [0.1234567]

    cnn = network([ &
      input(1, 8, 8), &
      conv2d(filters=1, kernel_size=3), &
      maxpool2d(pool_size=2), &
      conv2d(filters=1, kernel_size=3), &
      dense(1) &
    ])

    do n = 1, num_iterations
      call cnn % forward(x)
      call cnn % backward(y)
      call cnn % update(optimizer=sgd(learning_rate=1.))
      if (all(abs(cnn % predict(x) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'convolutional network 2 should converge in simple training.. failed'
      ok = .false.
    end if

  end block training2

  training3: block

    type(network) :: cnn
    real :: x(1, 12, 12)
    real :: y(9)
    real :: tolerance = 1e-5
    integer :: n
    integer, parameter :: num_iterations = 5000

    call random_number(x)
    y = [0.12345, 0.23456, 0.34567, 0.45678, 0.56789, 0.67890, 0.78901, 0.89012, 0.90123]

    cnn = network([ &
      input(1, 12, 12), &
      conv2d(filters=1, kernel_size=3), & ! 1x12x12 input, 1x10x10 output
      maxpool2d(pool_size=2), &           ! 1x10x10 input, 1x5x5 output
      conv2d(filters=1, kernel_size=3), & ! 1x5x5 input, 1x3x3 output
      dense(9) &                          ! 9 outputs
    ])

    do n = 1, num_iterations
      call cnn % forward(x)
      call cnn % backward(y)
      call cnn % update(optimizer=sgd(learning_rate=1.))
      if (all(abs(cnn % predict(x) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'convolutional network 3 should converge in simple training.. failed'
      ok = .false.
    end if

  end block training3


  if (ok) then
    print '(a)', 'test_conv2d_network: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv2d_network: One or more tests failed.'
    stop 1
  end if

end program test_conv2d_network
