submodule(nf_multihead_attention_layer) nf_multihead_attention_layer_submodule
  use nf_activation, only: softmax
  use nf_base_layer, only: base_layer
  use nf_linear2d_layer, only: linear2d_layer

  implicit none

contains
  module function multihead_attention_layer_cons(n_heads) result(res)
    integer, intent(in) :: n_heads
    type(multihead_attention_layer) :: res

    res % n_heads = n_heads
  end function multihead_attention_layer_cons

  pure module subroutine common_backward(self, input, gradient, attention_mask)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: input(:, :)
    real, intent(in) :: gradient(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    integer :: head, seq, i, j

    self % v_heads = self % split_heads(self % value_layer % output)
    self % k_heads = self % split_heads(self % key_layer % output)
    self % q_heads = self % split_heads(self % query_layer % output)

    ! bakward through attention mechanism
    call self % sdpa_backward(gradient, attention_mask)

    ! calculate deltas for input layers
    call self % value_layer % backward(self % v_input, self % combine_heads(self % v_or_dv))
    call self % key_layer % backward(self % k_input, self % combine_heads(self % k_or_dk))
    call self % query_layer % backward(self % q_input, self % combine_heads(self % q_or_dq))
  end subroutine common_backward

  pure module subroutine common_forward(self, query, key, value, attention_mask)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: query(:, :), key(:, :), value(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    self % q_input = query
    self % k_input = key
    self % v_input = value

    ! run inputs through linear layers (trainable params)
    call self % query_layer % forward(query)
    call self % key_layer % forward(key)
    call self % value_layer % forward(value)

    ! split attention heads for more efficient computation
    self % q_or_dq = self % split_heads(self % query_layer % output)
    self % k_or_dk = self % split_heads(self % key_layer % output)
    self % v_or_dv = self % split_heads(self % value_layer % output)

    call self % sdpa_forward(attention_mask)
  end subroutine common_forward

  pure module subroutine sdpa_forward(self, attention_mask)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in), optional :: attention_mask(:, :)

    ! create key by value matrix
    call self % create_attention_matrix(self % q_or_dq, self % k_or_dk)
    ! apply softmax and scaling
    call self % normalize_attention_matrix(attention_mask)
    ! multiply attention matrix by value
    call self % scaled_dot_product_attention(self % v_or_dv)

    self % o_input = self % combine_heads(self % sdpa)
    call self % output_layer % forward(self % o_input)
    self % output = self % output_layer % output
  end subroutine sdpa_forward

  pure module subroutine sdpa_backward(self, gradient, attention_mask)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: gradient(:, :)
    real, intent(in), optional :: attention_mask(:, :)

    integer :: head, seq, i, j

    ! calculate output layer delta
    call self % output_layer % backward(self % o_input, gradient)

    ! split heads from output gradient
    self % d_output = self % split_heads(self % output_layer % gradient)

    ! iterate over heads to calculate deltas for each of them
    do concurrent(head = 1: self % n_heads)
      self % v_or_dv(:, :, head) = matmul(&
          transpose(self % attention_matrix(:, :, head)),&
          self % d_output(:, :, head)&
      )

      ! calculate delta for attention matrix
      self % d_sdpa = matmul(self % d_output(:, :, head), transpose(self % v_heads(:, :, head)))

      if (present(attention_mask)) then
        self % d_sdpa = self % d_sdpa + attention_mask
      end if

      ! this monstrosity below is scaled derivative of softmax
      do concurrent(seq = 1: self % sequence_length)
        ! create jacobian matrix
        do concurrent(i = 1: self % sequence_length, j = 1: self % sequence_length)
          ! jacobian matrix is used to calculate derivative of softmax (temporary storage)
          ! the idea behind this if-else is that for diagonal elements, the jacobian temp
          ! should be: `softmax(x_i) * (1 - softmax(x_i))`
          ! for off-diagonal: `-softmax(x_i) * softmax(x_j)`
          if (i == j) then
            self % jacobian(i, j) = &
                self % attention_matrix(seq, i, head) &
                * (1 - self % attention_matrix(seq, i, head))
          else
            self % jacobian(i, j) = &
                - self % attention_matrix(seq, i, head) &
                * self % attention_matrix(seq, j, head)
          end if
        end do
        ! attention normalization delta, the last step of softmax derivative:
        ! multiply output of softmax by temp jacobian matrix
        ! For computational efficiency (avoid more temp storages), scaling is also done here
        ! reshapes: [3] -> [1, 3] @ [3, 3] = [1, 3] -> [3]
        self % d_normalize(seq, :, head) = reshape(matmul(&
            reshape(self % d_sdpa(seq, :), [1, self % sequence_length]),&
            self % jacobian * self % scaling_factor&
        ), [self % sequence_length])
      end do

      ! calculate delta for query
      self % q_or_dq(:, :, head) = matmul(self % d_normalize(:, :, head), self % k_heads(:, :, head))

      ! calculate delta for key, attention matrix should be transposed unlike for query
      self % k_or_dk(:, :, head) = matmul(transpose(self % d_normalize(:, :, head)), self % q_heads(:, :, head))
    end do
  end subroutine sdpa_backward

  pure module function split_heads(self, input) result(output)
    class(multihead_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :)
    real :: output(self % sequence_length, self % head_size, self % n_heads)
    output = reshape(input, [self % sequence_length, self % head_size, self % n_heads])
  end function split_heads

  pure module subroutine create_attention_matrix(self, query, key)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: query(:, :, :)
    real, intent(in) :: key(:, :, :)
    integer :: head
    ! create attention matrix for each sequence in each batch
    do concurrent(head = 1: self % n_heads)
      self % attention_matrix(:, :, head) = matmul(query(:, :, head), transpose(key(:, :, head)))
    end do
  end subroutine create_attention_matrix

  pure module subroutine normalize_attention_matrix(self, attention_mask)
    class(multihead_attention_layer), intent(in out) :: self
    real, optional, intent(in) :: attention_mask(:, :)
    integer :: head, seq

    ! scale dowm by square root of each head's size
    self % attention_matrix = self % attention_matrix * self % scaling_factor
    ! attention mask is used to mask out some of the tokens if necessary
    if (present(attention_mask)) then
      do concurrent(head = 1: self % n_heads)
        self % attention_matrix(:, :, head) = self % attention_matrix(:, :, head) + attention_mask
      end do
    end if
    ! softmax by last sequnce_length
    do concurrent(head = 1: self % n_heads, seq = 1: self % sequence_length)
      self % normalized_attention(seq, :, head) = self % softmax_func % eval_1d(self % attention_matrix(seq, :, head))
    end do
    self % attention_matrix = self % normalized_attention
  end subroutine normalize_attention_matrix

  pure module subroutine scaled_dot_product_attention(self, value)
    class(multihead_attention_layer), intent(in out) :: self
    real, intent(in) :: value(:, :, :)
    integer :: head

    do concurrent(head = 1: self % n_heads)
      self % sdpa(:, :, head) = matmul(self % attention_matrix(:, :, head), value(:, :, head))
    end do
  end subroutine scaled_dot_product_attention

  pure module function combine_heads(self, input) result(output)
    class(multihead_attention_layer), intent(in) :: self
    real, intent(in) :: input(:, :, :)
    real :: output(self % sequence_length, self % model_dimension)
    integer :: seq

    do concurrent(seq = 1: self % sequence_length)
      output(seq, :) = reshape(transpose(input(seq, :, :)), [self % model_dimension])
    end do
  end function combine_heads

  elemental module function get_num_params(self) result(num_params)
    class(multihead_attention_layer), intent(in) :: self
    integer :: num_params

    num_params = &
        self % query_layer % get_num_params() &
        + self % key_layer % get_num_params() &
        + self % value_layer % get_num_params() &
        + self % output_layer % get_num_params()
  end function get_num_params

  module function get_params(self) result(params)
    class(multihead_attention_layer), intent(in) :: self
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
    class(multihead_attention_layer), intent(in) :: self
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
    real, intent(in) :: params(:)
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

  module subroutine init_base(self, input_shape)
    class(multihead_attention_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    if (size(input_shape) /= 2) then
      error stop "MultiHead Attention accepts 2D input"
    end if
    self % sequence_length = input_shape(1)
    self % model_dimension = input_shape(2)

    if (mod(self % model_dimension, self % n_heads) /= 0) then
      write(stderr, '(a)'), 'Number of heads must be divisible by model dimension'
      error stop
    end if
    self % head_size = self % model_dimension / self % n_heads
    self % softmax_func = softmax()

    self % query_layer = linear2d_layer(self % model_dimension)
    self % key_layer = linear2d_layer(self % model_dimension)
    self % value_layer = linear2d_layer(self % model_dimension)
    self % output_layer = linear2d_layer(self % model_dimension)
    call self % query_layer % init([self % sequence_length, self % model_dimension])
    call self % key_layer % init([self % sequence_length, self % model_dimension])
    call self % value_layer % init([self % sequence_length, self % model_dimension])
    call self % output_layer % init([self % sequence_length, self % model_dimension])

    allocate(self % attention_matrix(self % sequence_length, self % sequence_length, self % n_heads))
    allocate(self % sdpa(self % sequence_length, self % head_size, self % n_heads))
    allocate(self % output(self % sequence_length, self % model_dimension))

    self % scaling_factor = sqrt(1 / real(self % head_size))

    allocate(self % q_input, mold=self % output)
    allocate(self % k_input, mold=self % output)
    allocate(self % v_input, mold=self % output)
    allocate(self % o_input, mold=self % output)

    ! allocate temporary storages

    ! this one is for forward pass
    allocate(self % normalized_attention, mold=self % attention_matrix)

    ! the following three are used twice:
    ! Forward pass: As inputs after the corresponding linear layer and head reshape
    ! Backward pass: As deltas for each input array
    allocate(self % q_or_dq, mold=self % sdpa)
    allocate(self % k_or_dk, mold=self % sdpa)
    allocate(self % v_or_dv, mold=self % sdpa)

    ! the other seven below are for backward pass
    allocate(self % d_output, mold=self % sdpa)
    allocate(self % v_heads, mold=self % sdpa)
    allocate(self % k_heads, mold=self % sdpa)
    allocate(self % q_heads, mold=self % sdpa)
    allocate(self % d_sdpa(self % sequence_length, self % sequence_length))
    allocate(self % jacobian, mold=self % d_sdpa)
    allocate(self % d_normalize, mold=self % attention_matrix)
  end subroutine init_base
end submodule nf_multihead_attention_layer_submodule