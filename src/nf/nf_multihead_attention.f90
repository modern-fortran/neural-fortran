module nf_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: softmax
  use nf_base_layer, only: base_layer
  use nf_dense_layer, only: dense_layer

  implicit none

  private
  public :: multihead_attention_layer

  type, extends(base_layer) :: multihead_attention_layer

    !! Concrete implementation of a multihead attention layer type

    integer :: batch_size, sequence_length, model_dimension, n_heads, head_size

    type(dense_layer) :: query_layer
    type(dense_layer) :: key_layer
    type(dense_layer) :: value_layer
    type(dense_layer) :: output_layer

    type(softmax) :: softmax_func

!    real :: output(batch_size, sequence_length, model_dimension)

  contains

!    procedure :: backward
    procedure :: forward
    procedure :: split_heads
    procedure :: create_attention_matrix
    procedure :: normalize_attention_matrix
    procedure :: scaled_dot_product_attention
    procedure :: combine_heads
    procedure :: init

  end type multihead_attention_layer

  interface multihead_attention_layer
    module function multihead_attention_layer_cons(batch_size, sequence_length, model_dimension, n_heads) result(res)
      !! This function returns the `multihead_attention_layer` instance.
      integer, intent(in) :: batch_size, sequence_length, model_dimension, n_heads
      type(multihead_attention_layer) :: res
    end function multihead_attention_layer_cons
  end interface multihead_attention_layer

  interface

    pure module subroutine backward(self, input, gradient)
      !! Apply the backward gradient descent pass.
      !! Only weight and bias gradients are updated in this subroutine,
      !! while the weights and biases themselves are untouched.
      class(multihead_attention_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward

    pure module subroutine forward(self, query, key, value)
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: query(:, :, :, :), key(:, :, :, :), value(:, :, :, :)
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(multihead_attention_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

contains
  module function multihead_attention_layer_cons(&
      batch_size, sequence_length, model_dimension, n_heads) result(res)
    integer, intent(in) :: batch_size, sequence_length, model_dimension, n_heads
    type(multihead_attention_layer) :: res
    res % batch_size = batch_size
    res % sequence_length = sequence_length
    res % model_dimension = model_dimension
    res % n_heads = n_heads

    if (mod(model_dimension, n_heads) /= 0) then
      write(stderr, '(a)'), 'Number of heads must be divisible by model dimension'
      error stop
    end if
    res % head_size = model_dimension / n_heads

    res % query_layer = dense_layer(input_size=model_dimension, output_size=model_dimension)
    res % key_layer = dense_layer(input_size=model_dimension, output_size=model_dimension)
    res % value_layer = dense_layer(input_size=model_dimension, output_size=model_dimension)
    res % output_layer = dense_layer(input_size=model_dimension, output_size=model_dimension)

    res % softmax_func = softmax()
  end function multihead_attention_layer_cons

  pure module subroutine forward(self, query, key, value)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: query(:, :, :, :), key(:, :, :, :), value(:, :, :, :)
  end subroutine forward

  module function split_heads(self, input) result(output)
    !! Split inputs into heads
    !!
    !! Example with two heads:
    !! input (1, 3, 4):
    !! [[[0.  , 0.3 , 0.6 , 0.9 ],
    !!   [0.1 , 0.4 , 0.7 , 0.11],
    !!   [0.2 , 0.5 , 0.8 , 0.12]]]
    !! output (1, 2, 3, 2)
    !! [[[[0.  , 0.3 ],
    !     [0.1 , 0.4 ],
    !     [0.2 , 0.5 ]],
    !    [[0.6 , 0.9 ],
    !     [0.7 , 0.11],
    !     [0.8 , 0.12]]]]
    class(multihead_attention_layer) :: self
    real :: input(:, :, :)
    real :: output(self % batch_size, self % n_heads, self % sequence_length, self % head_size)
    output = reshape(&
      input,&
      [self % batch_size, self % n_heads, self % sequence_length, self % head_size],&
      order=[1, 3, 4, 2]&
    )
  end function split_heads

  module function create_attention_matrix(self, query, key) result(output)
    !! Create attention matrix for query and key
    class(multihead_attention_layer) :: self
    real :: query(:, :, :, :)
    real :: key(:, :, :, :)
    real :: output(self % batch_size, self % n_heads, self % sequence_length, self % sequence_length)
    integer :: i, j
    ! create attention matrix for each sequence in each batch
    do i = 1, size(query(:, 1, 1, 1))
      do j = 1, size(query(1, :, 1, 1))
        output(i, j, :, :) = matmul(query(i, j, :, :), transpose(key(i, j, :, :)))
      end do
    end do
  end function create_attention_matrix

  module function normalize_attention_matrix(self, attention_matrix, attention_mask) result(output)
    !! Create attention matrix for query and key
    class(multihead_attention_layer) :: self
    real :: attention_matrix(:, :, :, :)
    !! (batch_size, n_heads, sequence_length, sequence_length)
    real, optional :: attention_mask(:, :, :, :)
    !! (batch_size, n_heads, sequence_length, sequence_length)
    real :: output(self % batch_size, self % n_heads, self % sequence_length, self % sequence_length)
    integer :: i, j, k

    ! scale dowm by square root of each head's size
    attention_matrix = attention_matrix / sqrt(real(self % head_size))
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      attention_matrix = attention_matrix + attention_mask
    end if
    ! softmax by last dimension
    do i = 1, size(output, 1)
      do j = 1, size(output, 2)
        do k = 1, size(output, 3)
          output(i, j, k, :) = self % softmax_func % eval_1d(attention_matrix(i, j, k, :))
        end do
      end do
    end do
  end function normalize_attention_matrix

  module function scaled_dot_product_attention(self, attention_matrix, value) result(output)
    class(multihead_attention_layer) :: self
    real :: attention_matrix(:, :, :, :)
    real :: value(:, :, :, :)
    real :: output(self % batch_size, self % n_heads, self % sequence_length, self % head_size)
    integer :: i, j

    do i = 1, size(attention_matrix, 1)
      do j = 1, size(attention_matrix, 2)
        output(i, j, :, :) = matmul(attention_matrix(i, j, :, :), value(i, j, :, :))
      end do
    end do
  end function scaled_dot_product_attention

  module function combine_heads(self, input) result(output)
    class(multihead_attention_layer) :: self
    real :: input(:, :, :, :)
    !! (batch_size, n_heads, sequence_length, head_size)
    real :: output(self % batch_size, self  % sequence_length, self % model_dimension)
    !! (batch_size, sequence_length, model_dimension)

    real :: scaled_dp_att_reshaped(self % batch_size, self  % sequence_length, self % n_heads, self % head_size)
    integer :: i, j

    scaled_dp_att_reshaped = reshape(input, shape(scaled_dp_att_reshaped), order=[1, 4, 2, 3])
    do i = 1, size(scaled_dp_att_reshaped, 1)
      do j = 1, size(scaled_dp_att_reshaped, 2)
        output(i, j, :) = reshape(scaled_dp_att_reshaped(i, j, :, :), [4])
      end do
    end do
  end function combine_heads

  module subroutine init(self, input_shape)
    class(multihead_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

  end subroutine init
end module nf_multihead_attention_layer
