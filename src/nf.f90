module nf
  !! User API: everything an application needs to reference directly
  use nf_datasets_mnist, only: label_digits, load_mnist
  use nf_layer, only: layer
  use nf_layer_constructors, only: &
    conv2d, dense, flatten, input, maxpool2d, reshape
  use nf_network, only: network
  use nf_optimizers, only: sgd
  use nf_activation, only: activation_function, elu, exponential,  &
                           gaussian, linear, relu, leaky_relu,     &
                           sigmoid, softmax, softplus, step, tanhf
end module nf
