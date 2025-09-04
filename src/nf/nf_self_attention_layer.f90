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
    module procedure self_attention_layer_cons
  end interface self_attention_layer

contains
  function self_attention_layer_cons(n_heads) result(res)
    !! This function returns the `self_attention_layer` instance.
    integer, intent(in) :: n_heads
    type(self_attention_layer) :: res
    res % n_heads = n_heads
  end function self_attention_layer_cons

  pure subroutine backward(self, input, gradient, attention_mask)
    !! Self Attention back propagation
    !! Returns sum of Query, Key and Value gradients
    class(self_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    call self % common_backward(input, gradient, attention_mask)
    self % gradient = &
        self % query_layer % gradient &
        + self % key_layer % gradient &
        + self % value_layer % gradient
  end subroutine backward

  pure subroutine forward(self, input)
    !! Cross Attention forward propagation
    !! Passes input three times into MultiHead Attention
    !! Input Shape: (sequence_length, model_dimension)
    class(self_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)

    call self % common_forward(input, input, input)
  end subroutine forward

  subroutine init(self, input_shape)
    class(self_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    call self % init_base(input_shape)
    allocate(self % gradient(self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_self_attention_layer
