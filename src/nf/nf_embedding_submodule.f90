submodule(nf_embedding_layer) nf_embedding_layer_submodule
  use nf_base_layer, only: base_layer
  implicit none
contains
  module function embedding_layer_cons(&
      sequence_length, vocab_size, model_dimension&
  ) result(res)
    integer, intent(in) :: sequence_length, vocab_size, model_dimension
    type(embedding_layer) :: res

    res % vocab_size = vocab_size
    res % model_dimension = model_dimension
    res % sequence_length = sequence_length
  end function embedding_layer_cons

  module subroutine init(self, input_shape)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    allocate(self % output(self % sequence_length, self % model_dimension))
    allocate(self % gradient(self % sequence_length, self % vocab_size))

    allocate(self % weights(self % vocab_size, self % model_dimension))
    self % weights = 0.1

    allocate(self % dw(self % vocab_size, self % model_dimension))
    self % dw = 0.0
  end subroutine init

  pure module subroutine forward(self, input)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input(:)
    integer :: i

    do concurrent(i = 1: self % sequence_length)
      self % output(i, :) = self % weights(input(i), :)
    end do
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(embedding_layer), intent(in out) :: self
    integer, intent(in) :: input(:)
    real, intent(in) :: gradient(:)
    real :: db(self % model_dimension)
    real :: dw(self % vocab_size, self % model_dimension)
    integer :: i
  end subroutine backward

  pure module function get_num_params(self) result(num_params)
    class(embedding_layer), intent(in) :: self
    integer :: num_params

    ! Number of weigths times number of biases
    num_params = self % vocab_size * self % model_dimension + self % model_dimension

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
