program get_set_network_params
  use nf, only: dense, input, network
  implicit none
  type(network) :: net, net2
  real :: x(1), y(1)
  real, parameter :: pi = 4*atan(1.)
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  real :: xtest(test_size), ytest(test_size)
  real :: ypred1(test_size), ypred2(test_size)
  integer :: i, n, nparam
  real, allocatable :: parameters(:)

  print '("Getting and setting network parameters")'
  print '(60("="))'

  net = network([ &
    input(1), &
    dense(5), &
    dense(3), & ! only for testing purposes (this layer is not really needed to solve this problem)
    dense(1) &
  ])

  call net % print_info()

  xtest = [((i - 1)*2*pi/test_size, i=1, test_size)]
  ytest = (sin(xtest) + 1)/2

  do n = 0, num_iterations

    call random_number(x)
    x = x*2*pi
    y = (sin(x) + 1)/2

    call net % forward(x)
    call net % backward(y)
    call net % update(1.)

    if (mod(n, 10000) == 0) then
      ypred1 = [(net % predict([xtest(i)]), i=1, test_size)]
      print '(i0,1x,f9.6)', n, sum((ypred1 - ytest)**2) / size(ypred1)
    end if

  end do

  print *, ''
  print '("Extract parameters")'
  print *, ''

  nparam = net % get_num_params()
  print '("get_num_params = ", i0)', nparam

  parameters = net % get_params()
  print '("size(parameters) = ", i0)', size(parameters)
  print *, 'parameters:', parameters

  print *, 'Now create another network of the same shape and set'
  print *, 'the parameters from the original network to it.'

  net2 = network([ &
    input(1), &
    dense(5), &
    dense(3), & ! only for testing purposes (this layer is not really needed to solve this problem)
    dense(1) &
  ])

  ! copy the parameters from net to net2
  call net2 % set_params(parameters)

  ypred1 = [(net % predict([xtest(i)]), i=1, test_size)]
  ypred2 = [(net2 % predict([xtest(i)]), i=1, test_size)]
  print *, 'Original and cloned network outputs match:', all(ypred1 == ypred2)

end program get_set_network_params
