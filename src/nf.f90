module nf
  !! User API: everything an application needs to reference directly
  use nf_datasets_mnist, only: label_digits, load_mnist
  use nf_layer, only: layer
  use nf_layer_constructors, only: &
    avgpool1d, &
    avgpool2d, &
    conv1d, &
    conv2d, &
    dense, &
    dropout, &
    embedding, &
    flatten, &
    input, &
    layernorm, &
    linear2d, &
    locally_connected1d, &
    maxpool1d, &
    maxpool2d, &
    reshape, &
    self_attention
  use nf_loss, only: mse, quadratic
  use nf_metrics, only: corr, maxabs
  use nf_network, only: network
  use nf_optimizers, only: sgd, rmsprop, adam, adagrad
  use nf_activation, only: activation_function, elu, exponential,  &
                           gaussian, linear, relu, leaky_relu,     &
                           sigmoid, softmax, softplus, step, tanhf, &
                           celu
  use nf_linear2d_layer, only: linear2d_layer
  use nf_multihead_attention_layer, only: multihead_attention_layer
end module nf
