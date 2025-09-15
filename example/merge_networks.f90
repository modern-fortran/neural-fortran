program merge_networks
  use nf, only: dense, input, network, sgd
  use nf_dense_layer, only: dense_layer
  implicit none

  type(network) :: net1, net2, net3
  real, allocatable :: x1(:), x2(:)
  real, allocatable :: y1(:), y2(:)
  real, allocatable :: y(:)
  integer, parameter :: num_iterations = 500
  integer :: n, nn
  integer :: net1_output_size, net2_output_size

  x1 = [0.1, 0.3, 0.5]
  x2 = [0.2, 0.4]
  y = [0.123456, 0.246802, 0.369258, 0.482604, 0.505050, 0.628406, 0.741852]

  net1 = network([ &
    input(3), &
    dense(2), &
    dense(3), &
    dense(2) &
  ])

  net2 = network([ &
    input(2), &
    dense(5), &
    dense(3) &
  ])

  net1_output_size = product(net1 % layers(size(net1 % layers)) % layer_shape)
  net2_output_size = product(net2 % layers(size(net2 % layers)) % layer_shape)

  ! Network 3
  net3 = network([ &
    input(net1_output_size + net2_output_size), &
    dense(7) &
  ])

  do n = 1, num_iterations

    ! Forward propagate two network branches
    call net1 % forward(x1)
    call net2 % forward(x2)

    ! Get outputs of net1 and net2, concatenate, and pass to net3
    ! A helper function could be made to take any number of networks
    ! and return the concatenated output. Such function would turn the following
    ! block into a one-liner.
    select type (net1_output_layer => net1 % layers(size(net1 % layers)) % p)
      type is (dense_layer)
        y1 = net1_output_layer % output
    end select

    select type (net2_output_layer => net2 % layers(size(net2 % layers)) % p)
      type is (dense_layer)
        y2 = net2_output_layer % output
    end select

    call net3 % forward([y1, y2])

    ! First compute the gradients on net3, then pass the gradients from the first
    ! hidden layer on net3 to net1 and net2, and compute their gradients.
    call net3 % backward(y)

    select type (next_layer => net3 % layers(2) % p)
      type is (dense_layer)
        call net1 % backward(y, gradient=next_layer % gradient(1:net1_output_size))
        call net2 % backward(y, gradient=next_layer % gradient(net1_output_size+1:size(next_layer % gradient)))
    end select

    ! Gradients are now computed on all networks and we can update the weights
    call net1 % update(optimizer=sgd(learning_rate=1.))
    call net2 % update(optimizer=sgd(learning_rate=1.))
    call net3 % update(optimizer=sgd(learning_rate=1.))

    if (mod(n, 50) == 0) then
      print *, "Iteration ", n, ", output RMSE = ", &
        sqrt(sum((net3 % predict([net1 % predict(x1), net2 % predict(x2)]) - y)**2) / size(y))
    end if

  end do

end program merge_networks