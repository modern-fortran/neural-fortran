submodule (nf_dropout_layer) nf_dropout_layer_submodule
  !! This submodule implements the procedures defined in the
  !! nf_dropout_layer module.

contains

  module function dropout_layer_cons(rate) result(res)
    real, intent(in) :: rate
    type(dropout_layer) :: res

    ! Initialize dropout rate
    res % dropout_rate = rate
  end function dropout_layer_cons


  module subroutine init(self, input_shape)
    class(dropout_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % input_size = input_shape(1)

    ! Allocate arrays
    allocate(self % output(self % input_size))
    allocate(self % gradient(self % input_size))
    allocate(self % mask(self % input_size))

    ! Initialize arrays
    self % output = 0
    self % gradient = 0
    self % mask = 1  ! Default mask is all ones (no dropout)

  end subroutine init


  module subroutine forward(self, input)
    class(dropout_layer), intent(in out) :: self
    real, intent(in) :: input(:)

    ! Generate random mask for dropout
    call random_number(self % mask)
    where (self % mask < self % dropout_rate)
      self % mask = 0
    elsewhere
      self % mask = 1
    end where

    ! Scale factor to preserve the input sum
    self % scale = sum(input) / sum(input * self % mask)

    ! Apply dropout mask
    self % output = input * self % mask * self % scale

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    class(dropout_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)

    ! Backpropagate gradient through dropout mask
    self % gradient = gradient * self % mask * self % scale
  end subroutine backward

end submodule nf_dropout_layer_submodule 