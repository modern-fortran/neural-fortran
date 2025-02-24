submodule(nf_fc2d_layer) nf_fc2d_layer_submodule
  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

contains
  module function fc2d_layer_cons(hidden_size, activation) result(res)
    !! This function returns the `fc2d_layer` instance.
    integer, intent(in) :: hidden_size
    class(activation_function), intent(in) :: activation
    type(fc2d_layer) :: res

    res % hidden_size = hidden_size
    res % activation_name = activation % get_name()
    ! FIXME: implement correct derivative for `softmax`
    if (res % activation_name == 'softmax') then
      write(stderr, '(a)') '`softmax` activation is temporarily unavailable'
      error stop 1
    end if
    allocate(res % activation, source = activation)
  end function fc2d_layer_cons

  module subroutine init(self, input_shape)
    class(fc2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "fc2d_layer accepts 2D input"
    end if

    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    self % in_proj = linear2d_layer(self % hidden_size)
    call self % in_proj % init([self % sequence_length, self % model_dimension])

    self % out_proj = linear2d_layer(self % model_dimension)
    call self % out_proj % init([self % sequence_length, self % hidden_size])

    allocate(self % in_proj_input(self % sequence_length, self % model_dimension))
    allocate(self % out_proj_input(self % sequence_length, self % hidden_size))

    allocate(self % output(self % sequence_length, self % model_dimension))

    allocate(self % gradient, mold=self % in_proj % gradient)
  end subroutine init

  pure module subroutine forward(self, input)
    class(fc2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    integer :: i

    self % in_proj_input = input
    call self % in_proj % forward(input)

    do concurrent(i = 1: self % sequence_length)
      self % out_proj_input(i, :) = self % activation % eval_1d(self % in_proj % output(i, :))
    end do

    call self % out_proj % forward(self % out_proj_input)
    self % output = self % out_proj % output
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(fc2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    integer :: i

    call self % out_proj % backward(self % out_proj_input, gradient)
    ! d_output/d_activation = d_output/d_output_proj * d/d_activation
    do concurrent(i = 1: self % sequence_length)
      self % out_proj % gradient(i, :) = &
          self % out_proj % gradient(i, :) &
          * (self % activation % eval_1d_prime(self % in_proj % output(i, :)))
    end do
    call self % in_proj % backward(self % in_proj_input, self % out_proj % gradient)

    self % gradient = self % in_proj % gradient
  end subroutine backward

  elemental module function get_num_params(self) result(num_params)
    class(fc2d_layer), intent(in) :: self
    integer :: num_params

    num_params = self % in_proj % get_num_params() + self % out_proj % get_num_params()
  end function get_num_params

  module function get_params(self) result(params)
    class(fc2d_layer), intent(in) :: self
    real, allocatable :: params(:)

    params = [&
        self % in_proj % weights,&
        self % out_proj % weights,&
        self % in_proj % biases,&
        self % out_proj % biases&
    ]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(fc2d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)

    gradients = [ &
        self % in_proj % dw,&
        self % out_proj % dw,&
        self % in_proj % db,&
        self % out_proj % db&
    ]
  end function get_gradients

  module subroutine set_params(self, params)
    class(fc2d_layer), intent(in out) :: self
    real, intent(in) :: params(:)

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    ! FIXME: looks clumsy, better ideas?
    associate (transformation => self % model_dimension * self % hidden_size)
      self % in_proj % weights = reshape(params(1: transformation), shape(self % in_proj % weights))
      self % out_proj % weights = reshape(&
          params(transformation + 1: 2 * transformation),&
          shape(self % out_proj % weights)&
      )
      self % in_proj % biases = params(2 * transformation + 1: 2 * transformation + self % hidden_size)
      self % out_proj % biases = params(&
          2 * transformation + self % hidden_size + 1: &
          2 * transformation + self % hidden_size + self % model_dimension&
      )
    end associate
  end subroutine set_params
end submodule nf_fc2d_layer_submodule
