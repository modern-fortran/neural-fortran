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
    module function cross_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
      !! This function returns the `cross_attention_layer` instance.
      integer, intent(in) :: sequence_length, model_dimension, n_heads
      type(cross_attention_layer) :: res
    end function cross_attention_layer_cons
  end interface cross_attention_layer

contains
  module function cross_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
    !! This function returns the `cross_attention_layer` instance.
    integer, intent(in) :: sequence_length, model_dimension, n_heads
    type(cross_attention_layer) :: res
    res % sequence_length = sequence_length
    res % model_dimension = model_dimension
    res % n_heads = n_heads

    if (mod(model_dimension, n_heads) /= 0) then
      write(stderr, '(a)'), 'Number of heads must be divisible by model dimension'
      error stop
    end if
    res % head_size = model_dimension / n_heads

    res % query_layer = linear2d_layer(model_dimension)
    res % key_layer = linear2d_layer(model_dimension)
    res % value_layer = linear2d_layer(model_dimension)
    res % output_layer = linear2d_layer(model_dimension)
    call res % query_layer % init([sequence_length, model_dimension])
    call res % key_layer % init([sequence_length, model_dimension])
    call res % value_layer % init([sequence_length, model_dimension])
    call res % output_layer % init([sequence_length, model_dimension])

    res % softmax_func = softmax()
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
