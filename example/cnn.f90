program cnn

  use nf, only: conv2d, dense, flatten, input, maxpool2d, network

  implicit none
  type(network) :: net
  real, allocatable :: x(:,:,:)
  integer :: n

  print '("Creating a CNN and doing a forward pass")'
  print '("(backward pass not implemented yet)")'
  print '(60("="))'

  net = network([ &
    input([3, 32, 32]), &
    conv2d(filters=16, kernel_size=3, activation='relu'), & ! (16, 30, 30)
    maxpool2d(pool_size=2), & ! (16, 15, 15)
    conv2d(filters=32, kernel_size=3, activation='relu'), & ! (32, 13, 13)
    maxpool2d(pool_size=2), & ! (32, 6, 6)
    flatten(), &
    dense(10) &
  ])

  ! Print a network summary to the screen
  call net % print_info()

  allocate(x(3,32,32))
  call random_number(x)

  print *, 'Output:', net % output(x)

end program cnn
