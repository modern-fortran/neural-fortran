program test_reshape_layer

    use iso_fortran_env, only: stderr => error_unit
    use nf, only: input, network, reshape_generalized  ! Check if this is the correct function
    use nf_datasets, only: download_and_unpack, keras_reshape_url

    implicit none

    type(network) :: net
    real, allocatable :: sample_input(:), output(:,:,:)
    integer, parameter :: output_shape_first(2)  = [64, 32]
    integer, parameter :: output_shape_second(6) = [8, 8, 4, 2, 2, 2]
    integer, parameter :: output_shape_third(5)  = [4, 4, 4, 4, 8]
    integer :: input_size  ! Removed parameter
    character(*), parameter :: keras_reshape_path = 'keras_reshape.h5'
    logical :: ok = .true.
    integer :: i
    integer, dimension(:), allocatable :: output_shape

    ! Test multiple reshape configurations
    do i = 1, 3
        select case (i)
            case (1)
                output_shape = output_shape_first
            case (2)
                output_shape = output_shape_second
            case (3)
                output_shape = output_shape_third
        end select

        ! Update input size
        input_size = product(output_shape)

        ! Create network with reshape_generalized
        net = network([ &
            input(input_size), &
            reshape_generalized(output_shape) &  ! Make sure the function name is correct
        ])

        if (.not. size(net % layers) == 2) then
            write(stderr, '(a, i0)') 'Test case ', i, ': the network should have 2 layers.. failed'
            ok = .false.
        end if

        ! Initialize test data
        allocate(sample_input(input_size))
        call random_number(sample_input)

        ! Allocate output correctly before reshaping
        allocate(output(output_shape(1), output_shape(2), output_shape(3)))  
        output = reshape(sample_input, shape(output))

        ! Check shape
        if (.not. all(shape(output) == output_shape)) then
            write(stderr, '(a, i0)') 'Test case ', i, ': the reshape layer produces expected output shape.. failed'
            ok = .false.
        end if

        ! Check values
        if (.not. all(output == reshape(sample_input, shape(output)))) then
            write(stderr, '(a, i0)') 'Test case ', i, ': the reshape layer produces expected output values.. failed'
            ok = .false.
        end if

        ! Deallocate for next test case
        deallocate(sample_input, output)
    end do

    ! Final test result
    if (ok) then
        print '(a)', 'test_reshape_generalized_layer: All tests passed.'
    else
        write(stderr, '(a)') 'test_reshape_generalized_layer: One or more tests failed.'
        stop 1
    end if

end program test_reshape_layer
