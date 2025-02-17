module nf_cross_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: softmax
  use nf_linear2d_layer, only: linear2d_layer
  use nf_multihead_attention_layer, only: multihead_attention_layer

  implicit none

  type, extends(multihead_attention_layer) :: cross_attention_layer
    !! Cross Attention Layer
    !! Source:
    !! Bahdanau, D. (2014)
    !! Neural machine translation by jointly learning to align and translate.
    !! https://arxiv.org/pdf/1409.0473
    real, allocatable :: gradient(:, :, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type cross_attention_layer

  interface cross_attention_layer
    module function cross_attention_layer_cons(n_heads) result(res)
      !! This function returns the `cross_attention_layer` instance.
      integer, intent(in) :: sequence_length, model_dimension, n_heads
      type(cross_attention_layer) :: res
    end function cross_attention_layer_cons
  end interface cross_attention_layer

contains
  module function cross_attention_layer_cons(n_heads) result(res)
    !! This function returns the `cross_attention_layer` instance.
    integer, intent(in) :: n_heads
    type(cross_attention_layer) :: res
    res % n_heads = n_heads
  end function cross_attention_layer_cons

  module subroutine backward(self, input, gradient)
    !! Cross Attention Back propagation
    class(cross_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :, :)
    real, intent(in) :: gradient(:, :)

    call self % common_backward(input(1, :, :), gradient)
    self % gradient(1, :, :) = self % query_layer % gradient
    self % gradient(2, :, :) = self % key_layer % gradient + self % value_layer % gradient
  end subroutine backward

  module subroutine forward(self, input)
    !! Cross Attention Forward propagation
    !! Input Shape (kind, sequence_length, model_dimension)
    !! where kind is 1 for Query and 2 for Key-Value
    class(cross_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :, :)

    call self % common_forward(input(1, :, :), input(2, :, :), input(2, :, :))
  end subroutine forward

  module subroutine init(self, input_shape)
    class(cross_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    call self % init_base(input_shape)
    allocate(self % gradient(2, self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_cross_attention_layer
