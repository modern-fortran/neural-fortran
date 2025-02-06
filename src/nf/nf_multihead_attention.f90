module nf_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_activation, only: softmax
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

  private
  public :: multihead_attention_layer

  type, extends(base_layer) :: multihead_attention_layer

    !! Concrete implementation of a multihead attention layer type

    integer :: batch_size, sequence_length, model_dimension, n_heads, head_size

    type(linear2d_layer) :: query_layer
    type(linear2d_layer) :: key_layer
    type(linear2d_layer) :: value_layer
    type(linear2d_layer) :: output_layer

    type(softmax) :: softmax_func

    real, allocatable :: attention_matrix(:, :, :, :)
    real, allocatable :: sdpa(:, :, :, :)
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

    module subroutine forward(self, query, key, value)
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

    res % query_layer = linear2d_layer(&
        sequence_length=sequence_length, in_features=model_dimension,&
        out_features=model_dimension, batch_size=batch_size&
      )
    res % key_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, batch_size)
    res % value_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, batch_size)
    res % output_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, batch_size)
    call res % query_layer % init([0])
    call res % key_layer % init([0])
    call res % value_layer % init([0])
    call res % output_layer % init([0])

    res % softmax_func = softmax()
  end function multihead_attention_layer_cons

  module subroutine forward(self, query, key, value)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: query(:, :, :), key(:, :, :), value(:, :, :)

    real :: q(self % n_heads, self % sequence_length, self % head_size, self % batch_size)
    real :: k(self % n_heads, self % sequence_length, self % head_size, self % batch_size)
    real :: v(self % n_heads, self % sequence_length, self % head_size, self % batch_size)
    real :: attention_matrix(self % n_heads, self % sequence_length, self % sequence_length, self % batch_size)
    real :: dot_product_attention(self % n_heads, self % sequence_length, self % head_size, self % batch_size)

    call self % query_layer % forward(query)
    call self % key_layer % forward(key)
    call self % value_layer % forward(value)

    q = self % split_heads(self % query_layer % output)
    k = self % split_heads(self % key_layer % output)
    v = self % split_heads(self % value_layer % output)

    call self % create_attention_matrix(q, k)
    call self % normalize_attention_matrix()
    call self % scaled_dot_product_attention(v)

    call self % output_layer % forward(self % combine_heads(self % sdpa))
    self % output = self % output_layer % output
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

  module subroutine create_attention_matrix(self, query, key)
    !! Create attention matrix for query and key
    !! Output dimensions: n_heads, sequence_length, sequence_length, batch_size
    class(multihead_attention_layer) :: self
    real :: query(:, :, :, :)
    real :: key(:, :, :, :)
    integer :: i, j
    ! create attention matrix for each sequence in each batch
    do i = 1, self % batch_size
      do j = 1, self % n_heads
        self % attention_matrix(j, :, :, i) = matmul(query(j, :, :, i), transpose(key(j, :, :, i)))
      end do
    end do
  end subroutine create_attention_matrix

  module subroutine normalize_attention_matrix(self, attention_mask)
    !! Create attention matrix for query and key
    !! Output dims: n_heads, sequence_length, sequence_length, batch_size
    class(multihead_attention_layer) :: self
    !! (batch_size, n_heads, sequence_length, sequence_length)
    real, optional :: attention_mask(:, :, :, :)
    !! (batch_size, n_heads, sequence_length, sequence_length)
    real, allocatable :: output(:, :, :, :)
    integer :: batch, head, seq

    ! temporary storage
    allocate(output(self % n_heads, self % sequence_length, self % sequence_length, self % batch_size))

    ! scale dowm by square root of each head's size
    self % attention_matrix = self % attention_matrix / sqrt(real(self % head_size))
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      self % attention_matrix = self % attention_matrix + attention_mask
    end if
    ! softmax by last sequnce_length
    do batch = 1, self % batch_size
      do head = 1, self % n_heads
        do seq = 1, self % sequence_length
          output(head, seq, :, batch) = self % softmax_func % eval_1d(&
              self % attention_matrix(head, seq, :, batch)&
          )
        end do
      end do
    end do
    self % attention_matrix = output

    deallocate(output)
  end subroutine normalize_attention_matrix

  module subroutine scaled_dot_product_attention(self, value)
    !! Create scaled dot product attention
    !! Output dims: n_heads, sequence_length, head_size, batch_size
    class(multihead_attention_layer) :: self
    real :: value(:, :, :, :)
    integer :: batch, head

    do batch = 1, self % batch_size
      do head = 1, self % n_heads
        self % sdpa(head, :, :, batch) = matmul(self % attention_matrix(head, :, :, batch), value(head, :, :, batch))
      end do
    end do
  end subroutine scaled_dot_product_attention

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

    allocate(self % attention_matrix(&
        self % n_heads, self % sequence_length, self % sequence_length, self % batch_size&
    ))
    allocate(self % sdpa(&
        self % n_heads, self % sequence_length, self % head_size, self % batch_size&
    ))
    allocate(self % output(self % sequence_length, self % model_dimension, self % batch_size))
  end subroutine init
end module nf_multihead_attention_layer
