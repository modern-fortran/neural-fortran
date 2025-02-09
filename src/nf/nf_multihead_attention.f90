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

    real :: scaling_factor

    real, allocatable :: q_input(:, :, :)
    real, allocatable :: k_input(:, :, :)
    real, allocatable :: v_input(:, :, :)
  contains

    procedure :: backward
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

    module subroutine backward(self, input, gradient)
      !! General backprop for MultiHead Attention mechanism
      !! Might be used for both Self and Cross Attention
      !! Self Attention: sum output gradients
      !! Cross Attention: use them separately
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: input(:, :, :)
      real, intent(in) :: gradient(:, :, :)
    end subroutine backward

    module subroutine forward(self, query, key, value)
      !! General forward propagation for MultiHead Attention Mechanism
      !! Might be used for both Self and Cross Attention
      !! Self Attention: pass the same value thrice
      !! Cross Attention: pass three values for your query, key and value
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: query(:, :, :), key(:, :, :), value(:, :, :)
    end subroutine forward

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(multihead_attention_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
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

  module subroutine backward(self, input, gradient)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :, :)
    real, intent(in) :: gradient(:, :, :)

    real, allocatable :: d_output(:, :, :, :)
    real, allocatable :: v_heads(:, :, :, :)
    real, allocatable :: k_heads(:, :, :, :)
    real, allocatable :: q_heads(:, :, :, :)
    real, allocatable :: dv(:, :, :, :)
    real, allocatable :: d_sdpa(:, :)
    real, allocatable :: jacobian(:, :)
    real, allocatable :: d_normalize(:, :, :, :)
    real, allocatable :: dq(:, :, :, :)
    real, allocatable :: dk(:, :, :, :)
    integer :: batch, head, seq, i, j

    ! allocate temporary storages for backward computation
    allocate(d_output(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(v_heads(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(k_heads(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(q_heads(self % n_heads, self % sequence_length, self % head_size, self % batch_size))

    allocate(dv(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(d_sdpa(self % sequence_length, self % sequence_length))
    allocate(jacobian(self % sequence_length, self % sequence_length))
    allocate(d_normalize(self % n_heads, self % sequence_length, self % sequence_length, self % batch_size))
    allocate(dq(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(dk(self % n_heads, self % sequence_length, self % head_size, self % batch_size))

    ! calculate output layer delta
    call self % output_layer % backward(input, gradient)

    ! split heads from output gradient
    d_output = self % split_heads(self % output_layer % gradient)
    v_heads = self % split_heads(self % value_layer % output)
    k_heads = self % split_heads(self % key_layer % output)
    q_heads = self % split_heads(self % query_layer % output)

    ! iterate over heads to calculate deltas for each of them
    do concurrent(batch = 1: self % batch_size, head = 1: self % n_heads)
      dv(head, :, :, batch) = matmul(transpose(self % attention_matrix(head, :, :, batch)), d_output(head, :, :, batch))

      ! calculate delta for attention matrix
      d_sdpa = matmul(d_output(head, :, :, batch), transpose(v_heads(head, :, :, batch)))

      ! this monstrosity below is scaled derivative of softmax
      do concurrent(seq = 1: self % sequence_length)
        ! create jacobian matrix
        do concurrent(i = 1: self % sequence_length, j = 1: self % sequence_length)
          ! jacobian matrix is used to calculate derivative of softmax (temporary storage)
          ! the idea behind this if-else is that for diagonal elements, the jacobian temp
          ! should be: `softmax(x_i) * (1 - softmax(x_i))`
          ! for off-diagonal: `-softmax(x_i) * softmax(x_j)`
          if (i == j) then
            jacobian(i, j) = &
                self % attention_matrix(head, seq, i, batch) &
                * (1 - self % attention_matrix(head, seq, i, batch))
          else
            jacobian(i, j) = &
                - self % attention_matrix(head, seq, i, batch) &
                * self % attention_matrix(head, seq, j, batch)
          end if
        end do
        ! attention normalization delta, the last step of softmax derivative:
        ! multiply output of softmax by temp jacobian matrix
        ! For computational efficiency (avoid more temp storages), scaling is also done here
        ! reshapes: [3] -> [1, 3] @ [3, 3] = [1, 3] -> [3]
        d_normalize(head, seq, :, batch) = reshape(matmul(&
            reshape(d_sdpa(seq, :), [1, self % sequence_length]),&
            jacobian * self % scaling_factor&
        ), [self % sequence_length])
      end do

      ! calculate delta for query
      dq(head, :, :, batch) = matmul(d_normalize(head, :, :, batch), k_heads(head, :, :, batch))

      ! calculate delta for key, attention matrix should be transposed unlike for query
      dk(head, :, :, batch) = matmul(transpose(d_normalize(head, :, :, batch)), q_heads(head, :, :, batch))
    end do

    ! calculate deltas for input layers
    call self % value_layer % backward(self % v_input, self % combine_heads(dv))
    call self % key_layer % backward(self % k_input, self % combine_heads(dk))
    call self % query_layer % backward(self % q_input, self % combine_heads(dq))

    ! free temporary storages
    deallocate(d_output)
    deallocate(v_heads)
    deallocate(k_heads)
    deallocate(q_heads)
    deallocate(d_sdpa)
    deallocate(jacobian)
    deallocate(d_normalize)
    deallocate(dq)
    deallocate(dk)
  end subroutine backward

  module subroutine forward(self, query, key, value)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: query(:, :, :), key(:, :, :), value(:, :, :)

    real, allocatable :: q(:, :, :, :)
    real, allocatable :: k(:, :, :, :)
    real, allocatable :: v(:, :, :, :)

    ! allocate storage for intermidiate stages
    allocate(q(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(k(self % n_heads, self % sequence_length, self % head_size, self % batch_size))
    allocate(v(self % n_heads, self % sequence_length, self % head_size, self % batch_size))

    self % q_input = query
    self % k_input = key
    self % v_input = value

    ! run inputs through linear layers (trainable params)
    call self % query_layer % forward(query)
    call self % key_layer % forward(key)
    call self % value_layer % forward(value)

    ! split attention heads for more efficient computation
    q = self % split_heads(self % query_layer % output)
    k = self % split_heads(self % key_layer % output)
    v = self % split_heads(self % value_layer % output)

    ! create key by value matrix
    call self % create_attention_matrix(q, k)
    ! apply softmax and scaling
    call self % normalize_attention_matrix()
    ! multiply attention matrix by value
    call self % scaled_dot_product_attention(v)

    call self % output_layer % forward(self % combine_heads(self % sdpa))
    self % output = self % output_layer % output

    ! free temp vars from memory
    deallocate(q)
    deallocate(k)
    deallocate(v)
  end subroutine forward

  module function split_heads(self, input) result(output)
    !! Split inputs into heads
    !!
    !! Example with two heads:
    !! input (3, 4, 1)
    !! output (2, 3, 2, 1)
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
    do concurrent(i = 1: self % batch_size, j = 1: self % n_heads)
      self % attention_matrix(j, :, :, i) = matmul(query(j, :, :, i), transpose(key(j, :, :, i)))
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
    self % attention_matrix = self % attention_matrix * self % scaling_factor
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      self % attention_matrix = self % attention_matrix + attention_mask
    end if
    ! softmax by last sequnce_length
    do concurrent(batch = 1: self % batch_size, head = 1: self % n_heads, seq = 1: self % sequence_length)
      output(head, seq, :, batch) = self % softmax_func % eval_1d(self % attention_matrix(head, seq, :, batch))
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

    do concurrent(batch = 1: self % batch_size, head = 1: self % n_heads)
      self % sdpa(head, :, :, batch) = matmul(self % attention_matrix(head, :, :, batch), value(head, :, :, batch))
    end do
  end subroutine scaled_dot_product_attention

  module function combine_heads(self, input) result(output)
    class(multihead_attention_layer) :: self
    real :: input(:, :, :, :)
    !! (n_heads, sequence_length, head_size, batch_size)
    real :: output(self % sequence_length, self % model_dimension, self % batch_size)
    integer :: batch, seq

    do concurrent(batch = 1: self % batch_size, seq = 1: self % sequence_length)
      output(seq, :, batch) = reshape(transpose(input(:, seq, :, batch)), [self % model_dimension])
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

    self % scaling_factor = sqrt(1 / real(self % head_size))

    allocate(self % q_input(self % sequence_length, self % model_dimension, self % batch_size))
    allocate(self % k_input(self % sequence_length, self % model_dimension, self % batch_size))
    allocate(self % v_input(self % sequence_length, self % model_dimension, self % batch_size))
  end subroutine init
end module nf_multihead_attention_layer
