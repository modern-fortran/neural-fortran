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

  module subroutine init(self, input_shape, training)
    class(dropout_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)
    logical, intent(in) :: training

    ! Set input and output sizes (dropout preserves dimensions)
    self % input_size = input_shape(1)
    self % output_size = input_shape(1)

    ! Allocate arrays
    if (allocated(self % output)) deallocate(self % output)
    if (allocated(self % gradient)) deallocate(self % gradient)
    if (allocated(self % mask)) deallocate(self % mask)

    allocate(self % output(self % output_size))
    allocate(self % gradient(self % input_size))
    allocate(self % mask(self % input_size))

    ! Initialize arrays to zero
    self % output = 0.0
    self % gradient = 0.0
    self % mask = 1.0  ! Default mask is all ones (no dropout)
  end subroutine init

  pure module subroutine forward(self, input)
    class(dropout_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real :: rand_vals(size(input))

    ! Generate random mask for dropout
    call random_number(rand_vals)
    where (rand_vals < self % dropout_rate)
      self % mask = 0
    elsewhere
      self % mask = 1 / (1 - self % dropout_rate)  ! Scale to preserve expected value
    end where

    ! Apply dropout mask
    self % output = input * self % mask
  end subroutine forward

  pure module subroutine backward(self, input, gradient)
    class(dropout_layer), intent(in out) :: self
    real, intent(in) :: input(:)
    real, intent(in) :: gradient(:)

    ! Backpropagate gradient through dropout mask
    self % gradient = gradient * self % mask
  end subroutine backward

end submodule nf_dropout_layer_submodule 