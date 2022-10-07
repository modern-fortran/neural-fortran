submodule(nf_dense_layer) nf_dense_layer_submodule

  use nf_activation_1d, only: activation_function, &
                              elu, elu_prime, &
                              exponential, &
                              gaussian, gaussian_prime, &
                              relu, relu_prime, &
                              sigmoid, sigmoid_prime, &
                              softmax, softmax_prime, &
                              softplus, softplus_prime, &
                              step, step_prime, &
                              tanhf, tanh_prime
  use nf_base_layer, only: base_layer
  use nf_random, only: randn

  implicit none

contains

  elemental module function dense_layer_cons(output_size, activation) &
    result(res)
    integer, intent(in) :: output_size
    character(*), intent(in) :: activation
    type(dense_layer) :: res
    res % output_size = output_size
    call res % set_activation(activation)
  end function dense_layer_cons


  pure module subroutine backward(self, input, gradient)
    class(dense_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)
    real :: db(self % output_size)
    real :: dw(self % input_size, self % output_size)

    db = gradient * self % activation_prime(self % z)
    dw = matmul(reshape(input, [size(input), 1]), reshape(db, [1, size(db)]))
    self % gradient = matmul(self % weights, db)
    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward


  pure module subroutine forward(self, input)
    class(dense_layer), intent(in out) :: self
    real, intent(in) :: input(:)

    self % z = matmul(input, self % weights) + self % biases
    self % output = self % activation(self % z)

  end subroutine forward


  module subroutine init(self, input_shape)
    class(dense_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_size = input_shape(1)

    ! Weights are a 2-d array of shape previous layer size
    ! times this layer size.
    allocate(self % weights(self % input_size, self % output_size))
    self % weights = randn(self % input_size, self % output_size) &
                   / self % input_size

    ! Broadcast weights to all other images, if any.
    call co_broadcast(self % weights, 1)

    allocate(self % biases(self % output_size))
    self % biases = 0

    allocate(self % output(self % output_size))
    self % output = 0

    allocate(self % z(self % output_size))
    self % z = 0

    allocate(self % dw(self % input_size, self % output_size))
    self % dw = 0

    allocate(self % db(self % output_size))
    self % db = 0

    allocate(self % gradient(self % output_size))
    self % gradient = 0

  end subroutine init


  elemental module subroutine set_activation(self, activation)
    class(dense_layer), intent(in out) :: self
    character(*), intent(in) :: activation

    select case(trim(activation))

      ! TODO need to figure out how to handle the alpha param
      !case('elu')
      !  self % activation => elu
      !  self % activation_prime => elu_prime
      !  self % activation_name = 'elu'

      case('exponential')
        self % activation => exponential
        self % activation_prime => exponential
        self % activation_name = 'exponential'

      case('gaussian')
        self % activation => gaussian
        self % activation_prime => gaussian_prime
        self % activation_name = 'gaussian'

      case('relu')
        self % activation => relu
        self % activation_prime => relu_prime
        self % activation_name = 'relu'

      case('sigmoid')
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
        self % activation_name = 'sigmoid'

      case('softmax')
        self % activation => softmax
        self % activation_prime => softmax_prime
        self % activation_name = 'softmax'

      case('softplus')
        self % activation => softplus
        self % activation_prime => softplus_prime
        self % activation_name = 'softplus'

      case('step')
        self % activation => step
        self % activation_prime => step_prime
        self % activation_name = 'step'

      case('tanh')
        self % activation => tanhf
        self % activation_prime => tanh_prime
        self % activation_name = 'tanh'

      case default
        error stop 'Activation must be one of: ' // &
          '"elu", "exponential", "gaussian", "relu", ' // &
          '"sigmoid", "softmax", "softplus", "step", ' // &
          'or "tanh".'

    end select

  end subroutine set_activation


  module subroutine update(self, learning_rate)
    class(dense_layer), intent(in out) :: self
    real, intent(in) :: learning_rate

    ! Sum weight and bias gradients across images, if any
    call co_sum(self % dw)
    call co_sum(self % db)

    self % weights = self % weights - learning_rate * self % dw
    self % biases = self % biases - learning_rate * self % db
    self % dw = 0
    self % db = 0

  end subroutine update

end submodule nf_dense_layer_submodule
