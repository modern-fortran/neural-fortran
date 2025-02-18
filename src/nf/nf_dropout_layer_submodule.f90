submodule (nf_dropout_layer) nf_dropout_layer_submodule
  use nf_random, only: shuffle
  !! This submodule implements the procedures defined in the
  !! nf_dropout_layer module.

contains

  module function dropout_layer_cons(rate) result(res)
    real, intent(in) :: rate
    type(dropout_layer) :: res
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

    ! Generate random mask for dropout, training mode only
    if (self % training) then

      ! Set the first dropout_rate number of elements to 0, the rest to 1,
      ! and shuffle. Note that the selection of the elements rounds down to
      ! the nearest integer, so in cases where size(input) * dropout_rate is
      ! not an integer, the actual dropout rate will be slightly lower.
      self % mask = 1
      self % mask(:int(size(self % mask) * self % dropout_rate)) = 0
      call shuffle(self % mask)

      ! Scale factor to preserve the input sum
      self % scale = sum(input) / sum(input * self % mask)

      ! Apply dropout mask
      self % output = input * self % mask * self % scale

    else
      ! In inference mode, we don't apply dropout; simply pass through the input
      self % output = input

    end if

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    class(dropout_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)

    if (self % training) then
      ! Backpropagate gradient through dropout mask
      self % gradient = gradient * self % mask * self % scale
    else
      ! In inference mode, pass through the gradient unchanged
      self % gradient = gradient
    end if
  end subroutine backward

end submodule nf_dropout_layer_submodule 