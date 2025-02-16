program test_dropout_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf, only: dense, dropout, input, layer, network
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

      if (layer1_p % dropout_rate /= 0.5) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer dropout rate should be 0.5.. failed'
      end if

      if (layer1_p % training) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer default training mode should be false.. failed'
      end if

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

  ! Test setting training mode explicitly.
  layer1 = dropout(0.5, training=.true.)
  select type(layer1_p => layer1 % p)
    type is(dropout_layer)
      if (.not. layer1_p % training) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer training mode should be true.. failed'
      end if
  end select

  layer1 = dropout(0.5, training=.false.)
  select type(layer1_p => layer1 % p)
    type is(dropout_layer)
      if (layer1_p % training) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer training mode should be false.. failed'
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

  ! Now we're gonna run the forward pass and check that the dropout indeed
  ! drops according to the requested dropout rate.
  forward_pass: block
    real :: input_data(5)
    real :: output_data(size(input_data))
    integer :: n

    net = network([ &
      input(size(input_data)), &
      dropout(0.5) &
    ])

    call random_number(input_data)
    do n = 1, 10000
      output_data = net % predict(input_data)
      ! Check that sum of output matches sum of input within small tolerance
      if (abs(sum(output_data) - sum(input_data)) > 1e-6) then
        ok = .false.
        exit
      end if
    end do
    if (.not. ok) then
      write(stderr, '(a)') 'dropout layer output sum should match input sum within tolerance.. failed'
    end if
  end block forward_pass


  training: block
    real :: x(10), y(5)
    real :: tolerance = 1e-3
    integer :: n
    integer, parameter :: num_iterations = 100000

    call random_number(x)
    y = [0.1234, 0.2345, 0.3456, 0.4567, 0.5678]

    net = network([ &
      input(10), &
      dropout(0.5, training=.true.), &
      dense(5) &
    ])

    do n = 1, num_iterations
      !select type(dropout_l => net % layers(2) % p)
      !  type is(dropout_layer)
      !    print *, dropout_l % training, dropout_l % mask
      !end select
      call net % forward(x)
      call net % backward(y)
      call net % update()
      !print *, n, net % predict(x)

      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'dense network should converge in simple training.. failed'
      ok = .false.
    end if

  end block training


  if (ok) then
    print '(a)', 'test_dropout_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_dropout_layer: One or more tests failed.'
    stop 1
  end if

end program test_dropout_layer
