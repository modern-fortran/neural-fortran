submodule(nf_embedding_layer) nf_embedding_layer_submodule
  use nf_base_layer, only: base_layer
  implicit none
contains
  module function embedding_layer_cons(vocab_size, model_dimension, positional) result(res)
    integer, intent(in) :: vocab_size, model_dimension
    logical, optional :: positional
    type(embedding_layer) :: res

    res % vocab_size = vocab_size
    res % model_dimension = model_dimension
    if (.not. present(positional)) then
      res % positional = .false.
    else
      res % positional = positional
    end if
  end function embedding_layer_cons

  module subroutine init(self, input_shape)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % sequence_length = input_shape(1)

    allocate(self % output(self % sequence_length, self % model_dimension))

    allocate(self % weights(self % vocab_size, self % model_dimension))
    self % weights = 0.1

    allocate(self % dw(self % vocab_size, self % model_dimension))
    self % dw = 0.0
  end subroutine init

  pure module subroutine forward(self, input)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input(:)
    integer :: i, index

    do concurrent(i = 1: self % sequence_length)
      index = input(i)
      if (index > size(self % weights, 1)) then
        index = 1
      elseif (index == 0) then
        index = 1
      end if

      self % output(i, :) = self % weights(index, :)

      if (self % positional) then
        call self % positional_encoding(i)
      end if
    end do
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input(:)
    real, intent(in) :: gradient(:, :)
    integer :: i

    do concurrent(i = 1: self % sequence_length)
      self % dw(input(i), :) = self % dw(input(i), :) + gradient(i, :)
    end do
  end subroutine backward

  pure module subroutine positional_encoding(self, pos)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: pos
    integer :: i
    real :: theta

    do concurrent(i = 1: floor(real(self % model_dimension) / 2))
      theta = (pos - 1) / 10000 ** (real(2 * (i-1)) / self % model_dimension)
      self % output(pos, 2 * i - 1) = self % output(pos, 2 * i - 1) + sin(theta)
      self % output(pos, 2 * i) = self % output(pos, 2 * i) + cos(theta)
    end do
  end subroutine positional_encoding

  pure module function get_num_params(self) result(num_params)
    class(embedding_layer), intent(in) :: self
    integer :: num_params
    num_params = self % vocab_size * self % model_dimension
  end function get_num_params

  module function get_params(self) result(params)
    class(embedding_layer), intent(in), target :: self
    real, allocatable :: params(:)
    real, pointer :: w_(:) => null()

    w_(1: product(shape(self % weights))) => self % weights
    params = [w_]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(embedding_layer), intent(in), target :: self
    real, allocatable :: gradients(:)
    real, pointer :: dw_(:) => null()

    dw_(1: product(shape(self % dw))) => self % dw
    gradients = [dw_]
  end function get_gradients

  module subroutine set_params(self, params)
    class(embedding_layer), intent(in out) :: self
    real, intent(in), target :: params(:)

    real, pointer :: p_(:,:) => null()

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    associate(n => self % vocab_size * self % model_dimension)
      ! reshape the weights
      p_(1:self % vocab_size, 1:self % model_dimension) => params(1 : n)
      self % weights = p_
    end associate

  end subroutine set_params
end submodule nf_embedding_layer_submodule
