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

    real, allocatable :: attention_matrix(:, :, :)
    real, allocatable :: sdpa(:, :, :)
    real, allocatable :: output(:, :)

    real :: scaling_factor

    real, allocatable :: q_input(:, :)
    real, allocatable :: k_input(:, :)
    real, allocatable :: v_input(:, :)
    real, allocatable :: o_input(:, :)
  contains

    procedure :: backward
    procedure :: forward
    procedure :: split_heads
    procedure :: create_attention_matrix
    procedure :: normalize_attention_matrix
    procedure :: scaled_dot_product_attention
    procedure :: combine_heads
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params
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
      real, intent(in) :: input(:, :)
      real, intent(in) :: gradient(:, :)
    end subroutine backward

    module subroutine forward(self, query, key, value)
      !! General forward propagation for MultiHead Attention Mechanism
      !! Might be used for both Self and Cross Attention
      !! Self Attention: pass the same value thrice
      !! Cross Attention: pass three values for your query, key and value
      class(multihead_attention_layer), intent(in out) :: self
      real, intent(in) :: query(:, :), key(:, :), value(:, :)
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
  module function multihead_attention_layer_cons(sequence_length, model_dimension, n_heads) result(res)
    integer, intent(in) :: sequence_length, model_dimension, n_heads
    type(multihead_attention_layer) :: res
    res % sequence_length = sequence_length
    res % model_dimension = model_dimension
    res % n_heads = n_heads

    if (mod(model_dimension, n_heads) /= 0) then
      write(stderr, '(a)'), 'Number of heads must be divisible by model dimension'
      error stop
    end if
    res % head_size = model_dimension / n_heads

    res % query_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, 1)
    res % key_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, 1)
    res % value_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, 1)
    res % output_layer = linear2d_layer(sequence_length, model_dimension, model_dimension, 1)
    call res % query_layer % init([0])
    call res % key_layer % init([0])
    call res % value_layer % init([0])
    call res % output_layer % init([0])

    res % softmax_func = softmax()
  end function multihead_attention_layer_cons

  module subroutine backward(self, input, gradient)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)

    real, allocatable :: d_output(:, :, :)
    real, allocatable :: v_heads(:, :, :)
    real, allocatable :: k_heads(:, :, :)
    real, allocatable :: q_heads(:, :, :)
    real, allocatable :: dv(:, :, :)
    real, allocatable :: d_sdpa(:, :)
    real, allocatable :: jacobian(:, :)
    real, allocatable :: d_normalize(:, :, :)
    real, allocatable :: dq(:, :, :)
    real, allocatable :: dk(:, :, :)
    integer :: head, seq, i, j

    ! allocate temporary storages for backward computation
    allocate(d_output(self % sequence_length, self % head_size, self % n_heads))
    allocate(v_heads(self % sequence_length, self % head_size, self % n_heads))
    allocate(k_heads(self % sequence_length, self % head_size, self % n_heads))
    allocate(q_heads(self % sequence_length, self % head_size, self % n_heads))

    allocate(dv(self % sequence_length, self % head_size, self % n_heads))
    allocate(d_sdpa(self % sequence_length, self % sequence_length))
    allocate(jacobian(self % sequence_length, self % sequence_length))
    allocate(d_normalize(self % sequence_length, self % sequence_length, self % n_heads))
    allocate(dq(self % sequence_length, self % head_size, self % n_heads))
    allocate(dk(self % sequence_length, self % head_size, self % n_heads))

    ! calculate output layer delta
    ! FIXME: remove reshapes when linear2d situation is resolved
    call self % output_layer % backward(&
        reshape(self % o_input, [self % sequence_length, self % model_dimension, 1]),&
        reshape(gradient, [self % sequence_length, self % model_dimension, 1])&
    )

    ! split heads from output gradient
    ! FIXME: remove reshapes when linear2d situation is resolved
    d_output = self % split_heads(&
        reshape(self % output_layer % gradient, [self % sequence_length, self % model_dimension]))
    v_heads = self % split_heads(&
        reshape(self % value_layer % output, [self % sequence_length, self % model_dimension]))
    k_heads = self % split_heads(reshape(self % key_layer % output, [self % sequence_length, self % model_dimension]))
    q_heads = self % split_heads(reshape(self % query_layer % output, [self % sequence_length, self % model_dimension]))

    ! iterate over heads to calculate deltas for each of them
    do concurrent(head = 1: self % n_heads)
      dv(:, :, head) = matmul(transpose(self % attention_matrix(:, :, head)), d_output(:, :, head))

      ! calculate delta for attention matrix
      d_sdpa = matmul(d_output(:, :, head), transpose(v_heads(:, :, head)))

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
                self % attention_matrix(seq, i, head) &
                * (1 - self % attention_matrix(seq, i, head))
          else
            jacobian(i, j) = &
                - self % attention_matrix(seq, i, head) &
                * self % attention_matrix(seq, j, head)
          end if
        end do
        ! attention normalization delta, the last step of softmax derivative:
        ! multiply output of softmax by temp jacobian matrix
        ! For computational efficiency (avoid more temp storages), scaling is also done here
        ! reshapes: [3] -> [1, 3] @ [3, 3] = [1, 3] -> [3]
        d_normalize(seq, :, head) = reshape(matmul(&
            reshape(d_sdpa(seq, :), [1, self % sequence_length]),&
            jacobian * self % scaling_factor&
        ), [self % sequence_length])
      end do

      ! calculate delta for query
      dq(:, :, head) = matmul(d_normalize(:, :, head), k_heads(:, :, head))

      ! calculate delta for key, attention matrix should be transposed unlike for query
      dk(:, :, head) = matmul(transpose(d_normalize(:, :, head)), q_heads(:, :, head))
    end do

    ! calculate deltas for input layers
    ! FIXME: remove reshapes when linear2d situation is resolved
    call self % value_layer % backward(&
        reshape(self % v_input, [self % sequence_length, self % model_dimension, 1]),&
        reshape(self % combine_heads(dv), [self % sequence_length, self % model_dimension, 1])&
    )
    call self % key_layer % backward(&
        reshape(self % k_input, [self % sequence_length, self % model_dimension, 1]),&
        reshape(self % combine_heads(dk), [self % sequence_length, self % model_dimension, 1])&
    )
    call self % query_layer % backward(&
        reshape(self % q_input, [self % sequence_length, self % model_dimension, 1]),&
        reshape(self % combine_heads(dq), [self % sequence_length, self % model_dimension, 1])&
    )

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
    real, intent(in) :: query(:, :), key(:, :), value(:, :)

    real, allocatable :: q(:, :, :)
    real, allocatable :: k(:, :, :)
    real, allocatable :: v(:, :, :)

    ! allocate storage for intermidiate stages
    allocate(q(self % sequence_length, self % head_size, self % n_heads))
    allocate(k(self % sequence_length, self % head_size, self % n_heads))
    allocate(v(self % sequence_length, self % head_size, self % n_heads))

    self % q_input = query
    self % k_input = key
    self % v_input = value

    ! run inputs through linear layers (trainable params)
    ! FIXME: remove reshapes when linear2d situation is resolved
    call self % query_layer % forward(reshape(query, [self % sequence_length, self % model_dimension, 1]))
    call self % key_layer % forward(reshape(key, [self % sequence_length, self % model_dimension, 1]))
    call self % value_layer % forward(reshape(value, [self % sequence_length, self % model_dimension, 1]))

    ! split attention heads for more efficient computation
    ! FIXME: remove reshapes when linear2d situation is resolved
    q = self % split_heads(reshape(self % query_layer % output, [self % sequence_length, self % model_dimension]))
    k = self % split_heads(reshape(self % key_layer % output, [self % sequence_length, self % model_dimension]))
    v = self % split_heads(reshape(self % value_layer % output,  [self % sequence_length, self % model_dimension]))

    ! create key by value matrix
    call self % create_attention_matrix(q, k)
    ! apply softmax and scaling
    call self % normalize_attention_matrix()
    ! multiply attention matrix by value
    call self % scaled_dot_product_attention(v)

    ! FIXME: remove reshapes when linear2d situation is resolved
    self % o_input = self % combine_heads(self % sdpa)
    call self % output_layer % forward(reshape(self % o_input, [self % sequence_length, self % model_dimension, 1]))
    self % output = reshape(self % output_layer % output, [self % sequence_length, self % model_dimension])

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
    real :: input(:, :)
    real :: output(self % sequence_length, self % head_size, self % n_heads)
    output = reshape(input, [self % sequence_length, self % head_size, self % n_heads])
  end function split_heads

  module subroutine create_attention_matrix(self, query, key)
    !! Create attention matrix for query and key
    !! Output dimensions: n_heads, sequence_length, sequence_length, batch_size
    class(multihead_attention_layer) :: self
    real :: query(:, :, :)
    real :: key(:, :, :)
    integer :: head
    ! create attention matrix for each sequence in each batch
    do concurrent(head = 1: self % n_heads)
      self % attention_matrix(:, :, head) = matmul(query(:, :, head), transpose(key(:, :, head)))
    end do
  end subroutine create_attention_matrix

  module subroutine normalize_attention_matrix(self, attention_mask)
    !! Create attention matrix for query and key
    !! Output dims: sequence_length, sequence_length, n_heads
    class(multihead_attention_layer) :: self
    !! (sequence_length, sequence_length, n_heads)
    real, optional :: attention_mask(:, :, :)
    !! (sequence_length, sequence_length, n_heads)
    real, allocatable :: output(:, :, :)
    integer :: head, seq

    ! temporary storage
    allocate(output(self % sequence_length, self % sequence_length, self % n_heads))

    ! scale dowm by square root of each head's size
    self % attention_matrix = self % attention_matrix * self % scaling_factor
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      self % attention_matrix = self % attention_matrix + attention_mask
    end if
    ! softmax by last sequnce_length
    do concurrent(head = 1: self % n_heads, seq = 1: self % sequence_length)
      output(seq, :, head) = self % softmax_func % eval_1d(self % attention_matrix(seq, :, head))
    end do
    self % attention_matrix = output

    deallocate(output)
  end subroutine normalize_attention_matrix

  module subroutine scaled_dot_product_attention(self, value)
    !! Create scaled dot product attention
    !! Output dims: n_heads, sequence_length, head_size, batch_size
    class(multihead_attention_layer) :: self
    real :: value(:, :, :)
    integer :: head

    do concurrent(head = 1: self % n_heads)
      self % sdpa(:, :, head) = matmul(self % attention_matrix(:, :, head), value(:, :, head))
    end do
  end subroutine scaled_dot_product_attention

  module function combine_heads(self, input) result(output)
    class(multihead_attention_layer) :: self
    real :: input(:, :, :)
    !! (sequence_length, head_size, n_heads)
    real :: output(self % sequence_length, self % model_dimension)
    integer :: seq

    do concurrent(seq = 1: self % sequence_length)
      output(seq, :) = reshape(transpose(input(seq, :, :)), [self % model_dimension])
    end do
  end function combine_heads

  module function get_num_params(self) result(num_params)
    class(multihead_attention_layer) :: self
    integer :: num_params

    num_params = &
        self % query_layer % get_num_params() &
        + self % key_layer % get_num_params() &
        + self % value_layer % get_num_params() &
        + self % output_layer % get_num_params()
  end function get_num_params

  module function get_params(self) result(params)
    class(multihead_attention_layer), intent(in), target :: self
    real, allocatable :: params(:)

    params = [&
        self % query_layer % weights,&
        self % key_layer % weights,&
        self % value_layer % weights,&
        self % output_layer % weights,&
        self % query_layer % biases,&
        self % key_layer % biases,&
        self % value_layer % biases,&
        self % output_layer % biases&
    ]
  end function get_params

  module function get_gradients(self) result(gradients)
    class(multihead_attention_layer), intent(in), target :: self
    real, allocatable :: gradients(:)

    gradients = [ &
        self % query_layer % dw,&
        self % key_layer % dw,&
        self % value_layer % dw,&
        self % output_layer % dw,&
        self % query_layer % db,&
        self % key_layer % db,&
        self % value_layer % db,&
        self % output_layer % db&
    ]
  end function get_gradients

  module subroutine set_params(self, params)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in), target :: params(:)
    real, pointer :: p_(:,:) => null()
    integer :: i, j, window

    ! check if the number of parameters is correct
    if (size(params) /= self % get_num_params()) then
      error stop 'Error: number of parameters does not match'
    end if

    ! FIXME: looks clumsy, better ideas?
    window = self % model_dimension * self % model_dimension
    i = 1
    j = window
    self % query_layer % weights = reshape(params(i: j), [self % model_dimension, self % model_dimension])
    i = j + 1
    j = i + window - 1
    self % key_layer % weights = reshape(params(i: j), [self % model_dimension, self % model_dimension])
    i = j + 1
    j = i + window - 1
    self % value_layer % weights = reshape(params(i: j), [self % model_dimension, self % model_dimension])
    i = j + 1
    j = i + window - 1
    self % output_layer % weights = reshape(params(i: j), [self % model_dimension, self % model_dimension])

    window = self % model_dimension
    i = j + 1
    j = i + window - 1
    self % query_layer % biases = params(i: j)
    i = j + 1
    j = i + window - 1
    self % key_layer % biases = params(i: j)
    i = j + 1
    j = i + window - 1
    self % value_layer % biases = params(i: j)
    i = j + 1
    j = i + window - 1
    self % output_layer % biases = params(i: j)
  end subroutine set_params

  module subroutine init(self, input_shape)
    class(multihead_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    allocate(self % attention_matrix(self % sequence_length, self % sequence_length, self % n_heads))
    allocate(self % sdpa(self % sequence_length, self % head_size, self % n_heads))
    allocate(self % output(self % sequence_length, self % model_dimension))

    self % scaling_factor = sqrt(1 / real(self % head_size))

    allocate(self % q_input(self % sequence_length, self % model_dimension))
    allocate(self % k_input(self % sequence_length, self % model_dimension))
    allocate(self % v_input(self % sequence_length, self % model_dimension))
    allocate(self % o_input(self % sequence_length, self % model_dimension))
  end subroutine init
end module nf_multihead_attention_layer
