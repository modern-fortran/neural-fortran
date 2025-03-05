program test_embedding_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_embedding_layer, only: embedding_layer
  use nf_layer, only: layer
  use nf_layer_constructors, only: embedding_constructor => embedding
  implicit none

  logical :: ok = .true.
  integer :: sample_input(3) = [2, 1, 3]

  call test_simple(ok, sample_input)
  call test_positional_trigonometric(ok, sample_input)
  call test_positional_absolute(ok, sample_input)

  if (ok) then
    print '(a)', 'test_embedding_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_embedding_layer: One or more tests failed.'
    error stop 1
  end if

contains
  subroutine test_simple(ok, sample_input)
    logical, intent(in out) :: ok
    integer, intent(in) :: sample_input(:)

    real :: sample_gradient(3, 2) = reshape([0.1, 0.2, 0.3, 0.4, 0.6, 0.6], [3, 2])
    real :: output_flat(6)
    real :: expected_output_flat(6) = reshape([0.3, 0.1, 0.5, 0.4, 0.2, 0.6], [6])
    real :: dw_flat(8)
    real :: expected_dw_flat(8) = reshape([0.2, 0.1, 0.3, 0., 0.6, 0.4, 0.6, 0.], [8])
    type(embedding_layer) :: embedding

    embedding = embedding_layer(vocab_size=4, model_dimension=2)
    call embedding % init([3])
    embedding % weights = reshape([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8], [4, 2])

    call embedding % forward(sample_input)

    output_flat = reshape(embedding % output, [6])
    if (.not. all(output_flat.eq.expected_output_flat)) then
      ok = .false.
      write(stderr, '(a)') 'forward returned incorrect values.. failed'
    end if

    call embedding % backward(sample_input, sample_gradient)
    dw_flat = reshape(embedding % dw, shape(dw_flat))
    if (.not. all(dw_flat.eq.expected_dw_flat)) then
      ok = .false.
      write(stderr, '(a)') 'backward returned incorrect dw values.. failed'
    end if
  end subroutine test_simple

  subroutine test_positional_trigonometric(ok, sample_input)
    logical, intent(in out) :: ok
    integer, intent(in) :: sample_input(:)

    real :: output_flat(12)
    real :: expected_output_flat(12) = reshape([&
        0.3, 0.941471, 1.4092975,&
        1.3, 0.64030236, 0.08385316,&
        0.3, 0.10999984, 0.51999867,&
        1.3, 1.09995, 1.4998&
    ], [12])
    type(embedding_layer) :: embedding

    real :: theta
    integer :: i, pos

    embedding = embedding_layer(vocab_size=5, model_dimension=4, positional=1)
    call embedding % init([3])
    embedding % weights = reshape([&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2&
    ], [5, 4])

    call embedding % forward(sample_input)

    output_flat = reshape(embedding % output, [12])
    if (.not. all(abs(output_flat - expected_output_flat) <= (1e-06 + 1e-05 * abs(expected_output_flat)))) then
      ok = .false.
      write(stderr, '(a)') 'trigonometric positional encoding returned incorrect values.. failed'
    end if
  end subroutine test_positional_trigonometric

  subroutine test_positional_absolute(ok, sample_input)
    logical, intent(in out) :: ok
    integer, intent(in) :: sample_input(:)

    real :: output_flat(12)
    real :: expected_output_flat(12) = reshape([&
        0.3, 1.1, 2.5,&
        0.3, 1.1, 2.5,&
        0.3, 1.1, 2.5,&
        0.3, 1.1, 2.5&
    ], [12])
    type(embedding_layer) :: embedding

    real :: theta
    integer :: i, pos

    embedding = embedding_layer(vocab_size=5, model_dimension=4, positional=2)
    call embedding % init([3])
    embedding % weights = reshape([&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2,&
        0.1, 0.3, 0.5, 0.7, 0.2&
    ], [5, 4])

    call embedding % forward(sample_input)

    output_flat = reshape(embedding % output, [12])
    if (.not. all(abs(output_flat - expected_output_flat) <= (1e-06 + 1e-05 * abs(expected_output_flat)))) then
      ok = .false.
      write(stderr, '(a)') 'absolute positional encoding returned incorrect values.. failed'
    end if
  end subroutine test_positional_absolute

  subroutine test_embedding_constructor(ok, sample_input)
    logical, intent(in out) :: ok
    integer, intent(in) :: sample_input(:)

    type(layer) :: embedding_constructed

    embedding_constructed = embedding_constructor(sequence_length=3, vocab_size=5, model_dimension=4)
    embedding_constructed = embedding_constructor(sequence_length=3, vocab_size=5, model_dimension=4, positional=0)
    embedding_constructed = embedding_constructor(sequence_length=3, vocab_size=5, model_dimension=4, positional=1)
    embedding_constructed = embedding_constructor(sequence_length=3, vocab_size=5, model_dimension=4, positional=2)
  end subroutine test_embedding_constructor
end program test_embedding_layer
