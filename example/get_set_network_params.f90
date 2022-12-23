program get_set_network_params
  use nf, only: dense, input, network
  implicit none
  type(network) :: net1, net2
  real :: x(1), y(1)
  real, parameter :: pi = 4 * atan(1.)
  integer, parameter :: num_iterations = 100000
  integer, parameter :: test_size = 30
  real :: xtest(test_size), ytest(test_size)
  real :: ypred1(test_size), ypred2(test_size)
  integer :: i, n, nparam
  real, allocatable :: parameters(:)

  print '("Getting and setting network parameters")'
  print '(60("="))'
  print *
  print '(a)', 'First, let''s instantiate small dense network net1'
  print '(a)', 'of shape (1,5,1) and fit it to a sine function:'
  print *

  net1 = network([ &
    input(1), &
    dense(5), &
    dense(1) &
  ])

  call net1 % print_info()

  xtest = [((i - 1) * 2 * pi / test_size, i=1, test_size)]
  ytest = (sin(xtest) + 1) / 2

  do n = 0, num_iterations

    call random_number(x)
    x = x * 2 * pi
    y = (sin(x) + 1) / 2

    call net1 % forward(x)
    call net1 % backward(y)
    call net1 % update(1.)

    if (mod(n, 10000) == 0) then
      ypred1 = [(net1 % predict([xtest(i)]), i=1, test_size)]
      print '(a,i0,1x,f9.6)', 'Number of iterations, loss: ', &
        n, sum((ypred1 - ytest)**2) / size(ypred1)
    end if

  end do

  print *
  print '(a)', 'Now, let''s see how many network parameters there are'
  print '(a)', 'by printing the result of net1 % get_num_params():'
  print *
  print '("net1 % get_num_params() = ", i0)', net1 % get_num_params()
  print *
  print '(a)', 'We can see the values of the network parameters'
  print '(a)', 'by printing the result of net1 % get_params():'
  print *
  print '("net1 % get_params() = ", *(g0,1x))', net1 % get_params()
  print *
  print '(a)', 'Now, let''s create another network of the same shape and set'
  print '(a)', 'the parameters from the original network to it'
  print '(a)', 'by calling call net2 % set_params(net1 % get_params()):'

  net2 = network([ &
    input(1), &
    dense(5), &
    dense(1) &
  ])

  ! Set the parameters of net1 to net2
  call net2 % set_params(net1 % get_params())

  print *
  print '(a)', 'We can check that the second network now has the same'
  print '(a)', 'parameters as net1:'
  print *
  print '("net2 % get_params() = ", *(g0,1x))', net2 % get_params()

  ypred1 = [(net1 % predict([xtest(i)]), i=1, test_size)]
  ypred2 = [(net2 % predict([xtest(i)]), i=1, test_size)]

  print *
  print '(a)', 'We can also check that the two networks produce the same output:'
  print *
  print '("net1 output: ", *(g0,1x))', ypred1
  print '("net2 output: ", *(g0,1x))', ypred2

  print *
  print '("Original and cloned network outputs match: ",l)', all(ypred1 == ypred2)

end program get_set_network_params
