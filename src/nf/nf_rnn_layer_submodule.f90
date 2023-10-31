submodule(nf_rnn_layer) nf_rnn_layer_submodule

  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_random, only: random_normal

  implicit none

contains

  elemental module function rnn_layer_cons(output_size, activation) &
    result(res)
    integer, intent(in) :: output_size
    class(activation_function), intent(in) :: activation
    type(rnn_layer) :: res

    res % output_size = output_size
    res % activation_name = activation % get_name()
    allocate( res % activation, source = activation )

  end function rnn_layer_cons


  pure module subroutine backward(self, input, gradient)
    class(rnn_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)
    real :: db(self % output_size)
    real :: dw(self % input_size, self % output_size)

    db = gradient * self % activation % eval_prime(self % z)
    dw = matmul(reshape(input, [size(input), 1]), reshape(db, [1, size(db)]))
    self % gradient = matmul(self % weights, db)
    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward


  pure module subroutine forward(self, input)
    class(rnn_layer), intent(in out) :: self
    real, intent(in) :: input(:)

    self % z = matmul(input, self % weights) &
               + matmul(self % state, self % recurrent) &
               + self % biases
    self % state = self % activation % eval(self % z)
    self % output = self % state

  end subroutine forward


  pure module function get_num_params(self) result(num_params)
    class(rnn_layer), intent(in) :: self
    integer :: num_params

    ! Number of weigths times number of biases
    num_params = self % input_size * self % output_size &
                 + self % output_size * self % output_size &
                 + self % output_size

  end function get_num_params


  pure module function get_params(self) result(params)
    class(rnn_layer), intent(in) :: self
    real, allocatable :: params(:)

    params = [ &
      pack(self % weights, .true.), &
      pack(self % recurrent, .true.), &
      pack(self % biases, .true.) &
    ]

  end function get_params


  pure module function get_gradients(self) result(gradients)
    class(rnn_layer), intent(in) :: self
    real, allocatable :: gradients(:)

    gradients = [ &
      pack(self % dw, .true.), &
      pack(self % db, .true.) &
    ]

  end function get_gradients


  module subroutine set_params(self, params)
    class(rnn_layer), intent(in out) :: self
    real, intent(in) :: params(:)
    integer :: first, last

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    ! reshape the weights
    last = self % input_size * self % output_size
    self % weights = reshape( &
      params(:last), &
      [self % input_size, self % output_size] &
    )

    ! reshape the recurrent weights
    first = last + 1
    last = first + self % output_size * self % output_size
    self % recurrent = reshape( &
      params(first:last), &
      [self % output_size, self % output_size] &
    )

    ! reshape the biases
    first = last + 1
    self % biases = reshape( &
      params(first:), &
      [self % output_size] &
    )

  end subroutine set_params


  module subroutine init(self, input_shape)
    class(rnn_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_size = input_shape(1)

    ! Weights are a 2-d array of shape previous layer size
    ! times this layer size.
    allocate(self % weights(self % input_size, self % output_size))
    call random_normal(self % weights)
    self % weights = self % weights / self % input_size

    ! Broadcast weights to all other images, if any.
    call co_broadcast(self % weights, 1)

    allocate(self % recurrent(self % output_size, self % output_size))
    call random_normal(self % recurrent)
    self % recurrent = self % recurrent / self % input_size


    allocate(self % biases(self % output_size))
    self % biases = 0

    allocate(self % output(self % output_size))
    self % output = 0

    allocate(self % z(self % output_size))
    self % z = 0

    allocate(self % state(self % output_size))
    self % state = 0

    allocate(self % dw(self % input_size, self % output_size))
    self % dw = 0

    allocate(self % db(self % output_size))
    self % db = 0

    allocate(self % gradient(self % output_size))
    self % gradient = 0

  end subroutine init

end submodule nf_rnn_layer_submodule
