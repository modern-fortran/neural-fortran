program test_locally_connected2d_layer

    use iso_fortran_env, only: stderr => error_unit
    use nf, only: locally_connected, input, layer
    use nf_input2d_layer, only: input2d_layer
  
    implicit none
  
    type(layer) :: locally_connected_1d_layer, input_layer
    integer, parameter :: filters = 32, kernel_size=3
    real, allocatable :: sample_input(:,:), output(:,:)
    real, parameter :: tolerance = 1e-7
    logical :: ok = .true.
  
    locally_connected_1d_layer = locally_connected(filters, kernel_size)
  
    if (.not. locally_connected_1d_layer % name == 'locally_connected2d') then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer has its name set correctly.. failed'
    end if
  
    if (locally_connected_1d_layer % initialized) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer should not be marked as initialized yet.. failed'
    end if
  
    if (.not. locally_connected_1d_layer % activation == 'relu') then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer defaults to relu activation.. failed'
    end if
  
    input_layer = input(3, 32)
    call locally_connected_1d_layer % init(input_layer)
  
    if (.not. locally_connected_1d_layer % initialized) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer should now be marked as initialized.. failed'
    end if
  
    if (.not. all(locally_connected_1d_layer % input_layer_shape == [3, 32])) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer input layer shape should be correct.. failed'
    end if
  
    if (.not. all(locally_connected_1d_layer % layer_shape == [filters, 30])) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer input layer shape should be correct.. failed'
    end if
  
    ! Minimal locally_connected_1d layer: 1 channel, 3x3 pixel image;
    allocate(sample_input(1, 3))
    sample_input = 0
  
    input_layer = input(1, 3)
    locally_connected_1d_layer = locally_connected(filters, kernel_size)
    call locally_connected_1d_layer % init(input_layer)
  
    select type(this_layer => input_layer % p); type is(input2d_layer)
      call this_layer % set(sample_input)
    end select
    deallocate(sample_input)
  
    call locally_connected_1d_layer % forward(input_layer)
    call locally_connected_1d_layer % get_output(output)
  
    if (.not. all(abs(output) < tolerance)) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer with zero input and sigmoid function must forward to all 0.5.. failed'
    end if
  
    ! Minimal locally_connected_1d layer: 1 channel, 3x3 pixel image, stride = 3;
    allocate(sample_input(1, 17))
    sample_input = 0
  
    input_layer = input(1, 17)
    locally_connected_1d_layer = locally_connected(filters, kernel_size, stride = 3)
    call locally_connected_1d_layer % init(input_layer)
  
    select type(this_layer => input_layer % p); type is(input2d_layer)
      call this_layer % set(sample_input)
    end select
    deallocate(sample_input)
  
    call locally_connected_1d_layer % forward(input_layer)
    call locally_connected_1d_layer % get_output(output)
  
    if (.not. all(abs(output) < tolerance)) then
      ok = .false.
      write(stderr, '(a)') 'locally_connected2d layer with zero input and sigmoid function must forward to all 0.5.. failed'
    end if

    !Final
    if (ok) then
      print '(a)', 'test_locally_connected2d_layer: All tests passed.'
    else
      write(stderr, '(a)') 'test_locally_connected2d_layer: One or more tests failed.'
      stop 1
    end if

end program test_locally_connected2d_layer
