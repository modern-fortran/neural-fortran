program test_conv3d_network

  use iso_fortran_env, only: stderr => error_unit
  use nf, only: conv, input, network, dense, sgd

  implicit none

  type(network) :: net
  real, allocatable :: sample_input(:,:,:,:), output(:,:,:,:)
  logical :: ok = .true.

  ! 3-layer convolutional network
  net = network([ &
    input(3, 8, 8, 8), &
    conv(filters=4, kernel_width=3, kernel_height=3, kernel_depth=3), &
    conv(filters=8, kernel_width=3, kernel_height=3, kernel_depth=3) &
  ])

  if (.not. size(net % layers) == 3) then
    write(stderr, '(a)') 'conv3d network should have 3 layers.. failed'
    ok = .false.
  end if

  ! Test for output shape
  allocate(sample_input(3, 8, 8, 8))
  sample_input = 0

  call net % forward(sample_input)
  call net % layers(3) % get_output(output)

  if (.not. all(shape(output) == [8, 4, 4, 4])) then
    write(stderr, '(a)') 'conv3d network output should have correct shape.. failed'
    ok = .false.
  end if

  deallocate(sample_input, output)

  training1: block

    type(network) :: cnn
    real :: y(1)
    real :: tolerance = 1e-4
    integer :: n
    integer, parameter :: num_iterations = 1000

    ! Test training of a minimal constant mapping
    allocate(sample_input(1, 5, 5, 5))
    call random_number(sample_input)

    cnn = network([ &
      input(1, 5, 5, 5), &
      conv(filters=1, kernel_width=3, kernel_height=3, kernel_depth=3), &
      conv(filters=1, kernel_width=3, kernel_height=3, kernel_depth=3), &
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
    real :: x(1, 7, 7, 7)
    real :: y(1)
    real :: tolerance = 1e-4
    integer :: n
    integer, parameter :: num_iterations = 1000

    call random_number(x)
    y = [0.1234567]

    cnn = network([ &
      input(1, 7, 7, 7), &
      conv(filters=1, kernel_width=3, kernel_height=3, kernel_depth=3), &
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

  if (ok) then
    print '(a)', 'test_conv3d_network: All tests passed.'
  else
    write(stderr, '(a)') 'test_conv3d_network: One or more tests failed.'
    stop 1
  end if

end program test_conv3d_network
