program test_multihead_attention_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_multihead_attention_layer, only: multihead_attention_layer
  implicit none

  logical :: ok = .true.
  type(multihead_attention_layer) :: attention
  real :: sample_input(1, 3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [1, 3, 4])
  real :: split_heads_output(1, 2, 3, 2)
  real :: raw_attention_matrix(1, 2, 3, 3)
  real :: normalized_attention_matrix(1, 2, 3, 3)

  attention = multihead_attention_layer(batch_size=1, sequence_length=3, model_dimension=4, n_heads=2)

  call test_multihead_attention_split_heads(attention, sample_input, ok, split_heads_output)
  call test_multihead_attention_create_attention_matrix(attention, split_heads_output, ok, raw_attention_matrix)
  call test_multihead_attention_normalization(attention, raw_attention_matrix, ok)

contains
  subroutine test_multihead_attention_split_heads(attention, input, ok, output)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :, :)
    logical, intent(in out) :: ok
    real, intent(in out) :: output(1, 2, 3, 2)
    real :: output_shape(4)
    real :: expected_shape(4) = [1, 2, 3, 2]
    real :: output_flat(12)
    real :: expected_output_flat(12) = [0.0, 0.6, 0.1, 0.7, 0.2, 0.8, 0.3, 0.9, 0.4, 0.11, 0.5, 0.12]

    output = attention % split_heads(input)

    output_shape = shape(output)
    if (.not. all(output_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'split_heads returned incorrect shape.. failed'
    end if
    output_flat = reshape(output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'split_heads returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_split_heads

  subroutine test_multihead_attention_create_attention_matrix(attention, input, ok, attention_matrix)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :, :, :)
    logical, intent(in out) :: ok
    real, intent(in out) :: attention_matrix(1, 2, 3, 3)
    real :: attention_matrix_shape(4)
    real :: attention_matrix_flat(18)
    real :: expected_shape(4) = [1, 2, 3, 3]
    real :: expected_attention_matrix_flat(18) = [&
        9.00000036E-02, 1.16999996, 0.120000005,&
        0.518999994, 0.150000006, 0.588000000,&
        0.120000005, 0.518999994, 0.170000002,&
        0.502099991, 0.219999999, 0.573199987,&
        0.150000006, 0.588000000, 0.219999999,&
        0.573199987, 0.289999992, 0.654400051&
    ]

    attention_matrix = attention % create_attention_matrix(input, input)

    attention_matrix_shape = shape(attention_matrix)
    if (.not. all(attention_matrix_shape.eq.expected_shape)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect shape.. failed'
    end if
    attention_matrix_flat = reshape(attention_matrix, shape(expected_attention_matrix_flat))
    if (.not. all(attention_matrix_flat.eq.expected_attention_matrix_flat)) then
      ok = .false.
      write(stderr, '(a)') 'create_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_create_attention_matrix

  subroutine test_multihead_attention_normalization(attention, input, ok)
    type(multihead_attention_layer), intent(in) :: attention
    real, intent(in) :: input(:, :, :, :)
    logical, intent(in out) :: ok
    real :: output(1, 2, 3, 3)
    real :: output_flat(18)
    real :: expected_output_flat(18) = [&
        0.326287806, 0.435975075, 0.321620107, 0.330339342, 0.316976935, 0.329200655,&
        0.333283335, 0.275134116, 0.333194464, 0.326415271, 0.333061278, 0.325773478,&
        0.340428889, 0.288890868, 0.345185399, 0.343245387, 0.349961787, 0.345025837&
    ]
    integer :: i, j, k
    real :: d_k, exp_x

    output = attention % normalize_attention_matrix(input)

    output_flat = reshape(output, shape(output_flat))
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'normalize_attention_matrix returned incorrect values.. failed'
    end if
  end subroutine test_multihead_attention_normalization
end program test_multihead_attention_layer