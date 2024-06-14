submodule(nf_dense_layer) nf_dense_layer_submodule

  use nf_optimizers, only: adam
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_random, only: random_normal

  implicit none

contains

  elemental module function dense_layer_cons(output_size, activation) &
    result(res)
    integer, intent(in) :: output_size
    class(activation_function), intent(in) :: activation
    type(dense_layer) :: res

    res % output_size = output_size
    res % activation_name = activation % get_name()
    allocate( res % activation, source = activation )

  end function dense_layer_cons


  pure module subroutine backward(self, input, gradient)
    class(dense_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)
    real :: db(self % output_size)
    real :: dw(self % input_size, self % output_size)
    integer :: i

    db = gradient * self % activation % eval_prime(self % z)
!    dw = matmul(reshape(input, [size(input), 1]), reshape(db, [1, size(db)]))
    do concurrent (i = 1:size(db))
      self % dw(:,i) = self % dw(:,i) + input(:) * db(i)
    enddo
    self % gradient = matmul(self % weights, db)
!    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward


  pure module subroutine forward(self, input)
    class(dense_layer), intent(in out) :: self
    real, intent(in) :: input(:)

    self % z = matmul(input, self % weights) + self % biases
    self % output = self % activation % eval(self % z)

  end subroutine forward


  pure module function get_num_params(self) result(num_params)
    class(dense_layer), intent(in) :: self
    integer :: num_params

    ! Number of weigths times number of biases
    num_params = self % input_size * self % output_size + self % output_size

  end function get_num_params


  module function get_params(self) result(params)
    class(dense_layer), intent(in), target :: self
    real, allocatable :: params(:)

    real, pointer :: w_(:) => null()

    w_(1:size(self % weights)) => self % weights

    params = [ &
      w_, &
      self % biases &
    ]

  end function get_params


  module function get_gradients(self) result(gradients)
    class(dense_layer), intent(in), target :: self
    real, allocatable :: gradients(:)

    real, pointer :: dw_(:) => null()

    dw_(1:size(self % dw)) => self % dw

    gradients = [ &
      dw_, &
      self % db &
    ]

  end function get_gradients


  module subroutine set_params(self, params)
    class(dense_layer), intent(in out) :: self
    real, intent(in), target :: params(:)

    real, pointer :: p_(:,:) => null()

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    associate(n => self % input_size * self % output_size)
      ! reshape the weights
      p_(1:self % input_size, 1:self % output_size) => params(1 : n)
      self % weights = p_

      ! reshape the biases
      self % biases = params(n + 1 : n + self % output_size)
    end associate

  end subroutine set_params


  module subroutine init(self, input_shape)
    class(dense_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_size = input_shape(1)

    ! Weights are a 2-d array of shape previous layer size
    ! times this layer size.
    allocate(self % weights(self % input_size, self % output_size))
    call random_normal(self % weights)
    self % weights = self % weights / self % input_size

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

    allocate(self % gradient(self % input_size))
    self % gradient = 0

  end subroutine init

  module subroutine set_optimizer(self, optimizer)
    class(dense_layer), intent(in out) :: self
    class(optimizer_base_type), intent(in), optional:: optimizer

    if (.not. allocated(self % optimizer_1d)) then
      if (present(optimizer)) then
        self % optimizer_1d = optimizer
      else
        self % optimizer_1d = adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1.e-7)
      end if
      call self % optimizer_1d % init(self % output_size)
    end if
    if (.not. allocated(self % optimizer_2d)) then
      if (present(optimizer)) then
        self % optimizer_2d = optimizer
      else
        self % optimizer_2d = adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1.e-7)
      end if
      call self % optimizer_2d % init(self % input_size * self % output_size)
    end if

  end subroutine set_optimizer

  module subroutine apply_optimizer(self, batch_size)
    class(dense_layer), intent(in out), target :: self
    integer, intent(in) :: batch_size

    real, pointer :: w_(:), dw_(:)

    call self % optimizer_1d % minimize( self % biases, self % db / batch_size)

    associate(n => self % input_size * self % output_size)
      w_(1:n) => self % weights
      dw_(1:n) => self % dw
      call self % optimizer_2d % minimize( w_, dw_ / batch_size)
    end associate


  end subroutine apply_optimizer

end submodule nf_dense_layer_submodule
