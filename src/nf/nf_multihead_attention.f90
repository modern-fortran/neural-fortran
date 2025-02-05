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

    real, allocatable :: output(:, :, :)

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
      real, intent(in) :: query(:, :, :), key(:, :, :), value(:, :, :)
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
    real, intent(in) :: query(:, :, :), key(:, :, :), value(:, :, :)

    real :: q(self % batch_size, self % n_heads, self % sequence_length, self % head_size)
    real :: k(self % batch_size, self % n_heads, self % sequence_length, self % head_size)
    real :: v(self % batch_size, self % n_heads, self % sequence_length, self % head_size)
    real :: attention_matrix(self % batch_size, self % n_heads, self % sequence_length, self % sequence_length)
    real :: dot_product_attention(self % batch_size, self % n_heads, self % sequence_length, self % head_size)

!    call self % query_layer % forward(query)
!    call self % key_layer % forward(key)
!    call self % value_layer % forward(value)
!
!    q = self % split_heads(self % query_layer % output)
!    k = self % split_heads(self % key_layer % output)
!    v = self % split_heads(self % value_layer % output)
!
!    attention_matrix = self % normalize_attention_matrix(self % create_attention_matrix(q, k))
!    dot_product_attention = self % scaled_dot_product_attention(attention_matrix, v)
!
!    self % output = self % output_layer % forward(self % combine_heads(dot_product_attention))
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
    real :: output(self % n_heads, self % sequence_length, self % head_size, self % batch_size)
    ! FIXME: if anybody knows how to also swap first two dims in one go, pls tell me
    output = reshape(&
      input,&
      [self % n_heads, self % sequence_length, self % head_size, self % batch_size],&
      order=[2, 4, 3, 1]&
    )
  end function split_heads

  module function create_attention_matrix(self, query, key) result(output)
    !! Create attention matrix for query and key
    class(multihead_attention_layer) :: self
    real :: query(:, :, :, :)
    real :: key(:, :, :, :)
    real :: output(self % n_heads, self % sequence_length, self % sequence_length, self % batch_size)
    integer :: i, j
    ! create attention matrix for each sequence in each batch
    do i = 1, self % batch_size
      do j = 1, self % n_heads
        output(j, :, :, i) = matmul(query(j, :, :, i), transpose(key(j, :, :, i)))
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
    real :: output(self % n_heads, self % sequence_length, self % sequence_length, self % batch_size)
    integer :: batch, head, seq

    ! scale dowm by square root of each head's size
    attention_matrix = attention_matrix / sqrt(real(self % head_size))
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      attention_matrix = attention_matrix + attention_mask
    end if
    ! softmax by second sequnce_length
    do batch = 1, self % batch_size
      do head = 1, self % n_heads
        do seq = 1, self % sequence_length
          output(head, seq, :, batch) = self % softmax_func % eval_1d(attention_matrix(head, seq, :, batch))
        end do
      end do
    end do
  end function normalize_attention_matrix

  module function scaled_dot_product_attention(self, attention_matrix, value) result(output)
    class(multihead_attention_layer) :: self
    real :: attention_matrix(:, :, :, :)
    real :: value(:, :, :, :)
    real :: output(self % n_heads, self % sequence_length, self % head_size, self % batch_size)
    integer :: batch, head

    do batch = 1, self % batch_size
      do head = 1, self % n_heads
        output(head, :, :, batch) = matmul(attention_matrix(head, :, :, batch), value(head, :, :, batch))
      end do
    end do
  end function scaled_dot_product_attention

  module function combine_heads(self, input) result(output)
    class(multihead_attention_layer) :: self
    real :: input(:, :, :, :)
    !! (n_heads, sequence_length, head_size, batch_size)
    real :: output(self % sequence_length, self % model_dimension, self % batch_size)
    integer :: batch, seq

    do batch = 1, self % batch_size
      do seq = 1, self % sequence_length
        output(seq, :, batch) = reshape(&
            transpose(input(:, seq, :, batch)), [self % model_dimension]&
        )
      end do
    end do
  end function combine_heads

  module subroutine init(self, input_shape)
    class(multihead_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    allocate(self % output(self % batch_size, self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_multihead_attention_layer
