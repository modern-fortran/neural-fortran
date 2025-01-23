program test_dropout_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dropout, layer
  use nf_dropout_layer, only: dropout_layer
  type(layer) :: layer1
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

  if (ok) then
    print '(a)', 'test_dropout_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_dropout_layer: One or more tests failed.'
    stop 1
  end if

end program test_dropout_layer
