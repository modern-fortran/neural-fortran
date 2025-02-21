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

      if (.not. layer1_p % training) then
        ok = .false.
        write(stderr, '(a)') 'dropout layer default training mode should be true.. failed'
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

  ! Test that the generated dropout mask matches the requested dropout rate.
  test_mask: block
    integer, parameter :: input_sizes(3) = [10, 100, 1000]
    real, parameter :: dropout_rates(5) = [0., 0.2, 0.5, 0.8, 1.]
    real, allocatable :: input_data(:)
    integer :: i, j

    do i = 1, size(input_sizes)
      do j = 1, size(dropout_rates)

        net = network([ &
          input(input_sizes(i)), &
          dropout(dropout_rates(j)) &
        ])

        if (allocated(input_data)) deallocate(input_data)
        allocate(input_data(input_sizes(i)))
        call random_number(input_data)

        call net % forward(input_data)

        select type(layer1_p => net % layers(2) % p)
          type is(dropout_layer)
            if (abs(sum(layer1_p % mask) / size(layer1_p % mask) - (1 - dropout_rates(j))) > 1e-6) then
              ok = .false.
              write(stderr, '(a)') 'actual dropout rate is equal to requested.. failed'
            end if
        end select
      end do
    end do

  end block test_mask


  ! Now we're gonna run the forward pass and check that the dropout indeed
  ! drops according to the requested dropout rate.
  forward_pass: block
    real :: input_data(10)
    real :: output_data(size(input_data))
    real, parameter :: dropout_rate = 0.2
    real :: realized_dropout_rate
    integer :: n

    net = network([ &
      input(size(input_data)), &
      dropout(dropout_rate) &
    ])

    do n = 1, 100

      call random_number(input_data)
      call net % forward(input_data)

      ! Check that sum of output matches sum of input within small tolerance
      select type(layer1_p => net % layers(2) % p)
        type is(dropout_layer)
          realized_dropout_rate = 1 - sum(input_data * layer1_p % mask) / sum(layer1_p % output)
          if (abs(realized_dropout_rate - dropout_rate) > 1e-6) then
            ok = .false.
            write(stderr, '(a)') 'realized dropout rate does not match requested dropout rate.. failed'
          end if
      end select

    end do

    if (.not. ok) write(stderr, '(a)') &
      'dropout layer output sum should match input sum within tolerance.. failed'

  end block forward_pass


  training: block
    real :: x(20), y(5)
    real :: tolerance = 1e-4
    integer :: n
    integer, parameter :: num_iterations = 100000

    call random_number(x)
    y = [0.12345, 0.23456, 0.34567, 0.45678, 0.56789]

    net = network([ &
      input(20), &
      dense(20), &
      dropout(0.2), &
      dense(5) &
    ])

    do n = 1, num_iterations
      call net % forward(x)
      call net % backward(y)
      call net % update()
      if (all(abs(net % predict(x) - y) < tolerance)) exit
    end do

    if (.not. n <= num_iterations) then
      write(stderr, '(a)') &
        'dense network should converge in simple training.. failed'
      ok = .false.
    end if

  end block training

  ! The following timing test is not part of the unit tests, but it's a good
  ! way to see the performance difference between a network with and without
  ! dropout.
  timing: block
    integer, parameter :: layer_size = 100
    integer, parameter :: num_iterations = 1000
    real :: x(layer_size), y(layer_size)
    integer :: n
    type(network) :: net1, net2
    real :: t1, t2
    real :: accumulated_time1 = 0
    real :: accumulated_time2 = 0

    net1 = network([ &
      input(layer_size), &
      dense(layer_size), &
      dense(layer_size) &
    ])

    net2 = network([ &
      input(layer_size), &
      dense(layer_size), &
      dropout(0.5), &
      dense(layer_size) &
    ])

    call random_number(y)

    ! Network without dropout
    do n = 1, num_iterations
      call random_number(x)
      call cpu_time(t1)
      call net1 % forward(x)
      call net1 % backward(y)
      call net1 % update()
      call cpu_time(t2)
      accumulated_time1 = accumulated_time1 + (t2 - t1)
    end do

    ! Network with dropout
    do n = 1, num_iterations
      call random_number(x)
      call cpu_time(t1)
      call net2 % forward(x)
      call net2 % backward(y)
      call net2 % update()
      call cpu_time(t2)
      accumulated_time2 = accumulated_time2 + (t2 - t1)
    end do

    ! Uncomment the following prints to see the timing results.
    !print '(a, f9.6, a, f9.6, a)', 'No dropout time: ', accumulated_time1, ' seconds'
    !print '(a, f9.6, a, f9.6, a)', 'Dropout time: ', accumulated_time2, ' seconds'

  end block timing

  if (ok) then
    print '(a)', 'test_dropout_layer: All tests passed.'
  else
    write(stderr, '(a)') 'test_dropout_layer: One or more tests failed.'
    stop 1
  end if

end program test_dropout_layer
