submodule(nf_dense_layer) nf_dense_layer_submodule

  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_random, only: random_normal

  implicit none

contains

  module function dense_layer_cons(output_size, activation) &
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


  module subroutine get_params_ptr(self, w_ptr, b_ptr)
    class(dense_layer), intent(in), target :: self
    real, pointer, intent(out) :: w_ptr(:)
    real, pointer, intent(out) :: b_ptr(:)
    w_ptr(1:size(self % weights)) => self % weights
    b_ptr => self % biases
  end subroutine get_params_ptr


  module subroutine get_gradients_ptr(self, dw_ptr, db_ptr)
    class(dense_layer), intent(in), target :: self
    real, pointer, intent(out) :: dw_ptr(:)
    real, pointer, intent(out) :: db_ptr(:)
    dw_ptr(1:size(self % dw)) => self % dw
    db_ptr => self % db
  end subroutine get_gradients_ptr


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
#ifdef PARALLEL
    call co_broadcast(self % weights, 1)
#endif

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

end submodule nf_dense_layer_submodule
