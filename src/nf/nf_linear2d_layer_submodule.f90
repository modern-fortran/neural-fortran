submodule(nf_linear2d_layer) nf_linear2d_layer_submodule
  use nf_base_layer, only: base_layer
  use nf_random, only: random_normal
  implicit none

contains

  module function linear2d_layer_cons(out_features, biases) result(res)
    integer, intent(in) :: out_features
    logical, optional, intent(in) :: biases
    type(linear2d_layer) :: res

    res % out_features = out_features
    if (present(biases)) then
      res % use_biases = biases
    else
      res % use_biases = .true.
    end if

  end function linear2d_layer_cons


  module subroutine init(self, input_shape)
    class(linear2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "linear2d layer requires 2D input."
    end if
    self % sequence_length = input_shape(1)
    self % in_features = input_shape(2)

    allocate(self % output(self % sequence_length, self % out_features))
    allocate(self % gradient(self % sequence_length, self % in_features))

    allocate(self % weights(self % in_features, self % out_features))
    call random_normal(self % weights)

    allocate(self % biases(self % out_features))
    call random_normal(self % biases)

    allocate(self % dw(self % in_features, self % out_features))
    self % dw = 0

    if (self % use_biases) then
      allocate(self % db(self % out_features))
      self % db = 0
    end if

  end subroutine init


  pure module subroutine forward(self, input)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    integer :: i

    self % output(:,:) = matmul(input(:,:), self % weights)
    if (self % use_biases) then
      do concurrent(i = 1:self % sequence_length)
        self % output(i,:) = self % output(i,:) + self % biases
      end do
    end if

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:)
    real, intent(in) :: gradient(:,:)
    real :: db(self % out_features)
    real :: dw(self % in_features, self % out_features)
    integer :: i

    self % dw = self % dw + matmul(transpose(input(:,:)), gradient(:,:))
    if (self % use_biases) then
      self % db = self % db + sum(gradient(:,:), 1)
    end if
    self % gradient(:,:) = matmul(gradient(:,:), transpose(self % weights))
  end subroutine backward


  pure module function get_num_params(self) result(num_params)
    class(linear2d_layer), intent(in) :: self
    integer :: num_params

    ! Number of weights times number of biases
    num_params = self % in_features * self % out_features
    if (self % use_biases) then
      num_params = num_params + self % out_features
    end if

  end function get_num_params


  module function get_params(self) result(params)
    class(linear2d_layer), intent(in), target :: self
    real, allocatable :: params(:)

    real, pointer :: w_(:) => null()

    w_(1: product(shape(self % weights))) => self % weights

    if (self % use_biases) then
      params = [ &
        w_, &
        self % biases &
      ]
    else
      params = w_
    end if

  end function get_params


  module function get_gradients(self) result(gradients)
    class(linear2d_layer), intent(in), target :: self
    real, allocatable :: gradients(:)

    real, pointer :: dw_(:) => null()

    dw_(1: product(shape(self % dw))) => self % dw

    if (self % use_biases) then
      gradients = [ &
        dw_, &
        self % db &
      ]
    else
      gradients = dw_
    end if

  end function get_gradients


  module subroutine set_params(self, params)
    class(linear2d_layer), intent(in out) :: self
    real, intent(in), target :: params(:)

    real, pointer :: p_(:,:) => null()

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    associate(n => self % in_features * self % out_features)
      ! reshape the weights
      p_(1:self % in_features, 1:self % out_features) => params(1 : n)
      self % weights = p_

      if (self % use_biases) then
        ! reshape the biases
        self % biases = params(n + 1 : n + self % out_features)
      end if
    end associate

  end subroutine set_params

end submodule nf_linear2d_layer_submodule
