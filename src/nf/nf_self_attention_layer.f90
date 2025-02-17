module nf_self_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: softmax
  use nf_linear2d_layer, only: linear2d_layer
  use nf_multihead_attention_layer, only: multihead_attention_layer

  implicit none

  type, extends(multihead_attention_layer) :: self_attention_layer
    !! Self Attention Layer
    !! Source:
    !! Parikh, A. P., Taeckstroem, O., Das, D., & Uszkoreit, J. (2016)
    !! A decomposable attention model for natural language inference.
    !! https://arxiv.org/pdf/1606.01933
    real, allocatable :: gradient(:, :)
  contains
    procedure :: forward
    procedure :: backward
    procedure :: init
  end type self_attention_layer

  interface self_attention_layer
    module function self_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
      !! This function returns the `self_attention_layer` instance.
      integer, intent(in) :: sequence_length, model_dimension, n_heads
      type(self_attention_layer) :: res
    end function self_attention_layer_cons
  end interface self_attention_layer

contains
  module function self_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
    !! This function returns the `self_attention_layer` instance.
    integer, intent(in) :: sequence_length, model_dimension, n_heads
    type(self_attention_layer) :: res
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
  end function self_attention_layer_cons

  module subroutine backward(self, input, gradient)
    !! Self Attention back propagation
    !! Returns sum of Query, Key and Value gradients
    class(self_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)

    call self % common_backward(input, gradient)
    self % gradient = &
        self % query_layer % gradient &
        + self % key_layer % gradient &
        + self % value_layer % gradient
  end subroutine backward

  module subroutine forward(self, input)
    !! Cross Attention forward propagation
    !! Passes input three times into MultiHead Attention
    !! Input Shape: (sequence_length, model_dimension)
    class(self_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)

    call self % common_forward(input, input, input)
  end subroutine forward

  module subroutine init(self, input_shape)
    class(self_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    call self % init_base(input_shape)
    allocate(self % gradient(self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_self_attention_layer
