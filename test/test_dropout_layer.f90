program test_dropout_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dropout, input, layer, network
  use nf_dropout_layer, only: dropout_layer
  type(layer) :: layer1
  type(network) :: net
  integer :: input_size

  logical :: ok = .true.

  layer1 = dropout(0.5)

  if (.not. layer1 % name == 'dropout') then
    ok = .false.
    write(stderr, '(a)') 'dropout layer has its name set correctly.. failed'
  end if

  ! Dropout on its own is not initialized and its arrays not allocated.
  select type(layer1_p => layer1 % p)
    type is(dropout_layer)

      if (layer1_p % input_size /= 0) then
        print *, 'input_size: ', layer1_p % input_size
        ok = .false.
        write(stderr, '(a)') 'dropout layer size should be zero.. failed'
      end if

      if (allocated(layer1_p % output)) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer output array should not be allocated.. failed'
      end if

  end select

  ! Now we're gonna initialize a minimal network with an input layer and a
  ! dropout that follows and we'll check that the dropout layer has expected
  ! state.
  input_size = 10
  net = network([ &
    input(input_size), &
    dropout(0.5) &
  ])

  select type(layer1_p => net % layers(1) % p)
    type is(dropout_layer)
      if (layer1_p % input_size /= input_size) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer input size should be the same as the input layer.. failed'
      end if

      if (.not. allocated(layer1_p % output)) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer output array should be allocated.. failed'
      end if

      if (.not. allocated(layer1_p % gradient)) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer gradient array should be allocated.. failed'
      end if

      if (.not. allocated(layer1_p % mask)) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer mask array should be allocated.. failed'
      end if

  end select

  if (ok) then
    print '(a)', 'test_dropout_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_dropout_layer: One or more tests failed.'
    stop 1
  end if

end program test_dropout_layer
