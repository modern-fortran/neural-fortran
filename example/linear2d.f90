program linear2d_example

  use nf, only: input, network, sgd, linear2d, mse, flatten
  implicit none

  type(network) :: net
  type(mse) :: loss
  real :: x(3, 4) = reshape( &
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12, 0.13], &
    [3, 4])
  real :: y(3) = [0.12, 0.1, 0.2]
  real :: preds(3)
  real :: loss_value
  integer, parameter :: num_iterations = 500
  integer :: n
  
  net = network([ &
    input(3, 4), &
    linear2d(3, 1), &
    flatten() &
  ])
  
  call net % print_info()
  loss = mse()

  do n = 1, num_iterations
    call net % forward(x)
    call net % backward(y, loss)
    call net % update(optimizer=sgd(learning_rate=0.01))
    preds = net % predict(x)
    print '(i4,3(3x,f8.6))', n, preds
    loss_value = loss % eval (y, preds)
    if (loss_value < 0.01) then
      print *, 'Loss: ', loss_value
      return
    end if
  end do

end program linear2d_example