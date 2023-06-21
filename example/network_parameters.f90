program network_parameters
  ! This program demonstrates how to access network parameters (weights and
  ! biases) from the layers' internal data structures.
  use nf, only: dense, input, layer, network
  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer

  implicit none

  type(network) :: net
  integer :: n

  net = network([input(3), dense(5), dense(2)])

  do n = 1, size(net % layers)
    print *, "Layer ", n, "is " // net % layers(n) % name
    select type (this_layer => net % layers(n) % p)
      type is (dense_layer) 
        print *, "  with weights of shape", shape(this_layer % weights)
        print *, "  and ", size(this_layer % biases), " biases"
        print *, "Weights are:"
        print *, this_layer % weights
      type is (conv2d_layer)
        print *, "  with kernel of shape", shape(this_layer % kernel)
        print *, "   and ", size(this_layer % biases), " biases"
        print *, "Kernel is:"
        print *, this_layer % kernel
      class default
        print *, "  with no parameters"
    end select
  end do

end program network_parameters