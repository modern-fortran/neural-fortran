! Copyright (c) 2024-2025, The Regents of the University of California and Sourcery Institute
! Terms of use are as specified in LICENSE.txt

module linear_2d_layer_test_m
  use julienne_m, only : &
     test_t, test_description_t, test_diagnosis_t, test_result_t &
    ,operator(.equalsExpected.), operator(//), operator(.approximates.), operator(.within.), operator(.also.), operator(.all.)
  use nf_linear2d_layer, only: linear2d_layer
  implicit none

  type, extends(test_t) :: linear_2d_layer_test_t
  contains
    procedure, nopass :: subject
    procedure, nopass :: results
  end type

contains

  pure function subject() result(test_subject)
    character(len=:), allocatable :: test_subject
    test_subject = 'A linear_2d_layer'
  end function

  function results() result(test_results)
    type(linear_2d_layer_test_t) linear_2d_layer_test
    type(test_result_t), allocatable :: test_results(:)
    test_results = linear_2d_layer_test%run( & 
      [test_description_t('updating gradients', check_gradient_updates) &
    ])
  end function

  function check_gradient_updates() result(test_diagnosis)
    type(test_diagnosis_t) test_diagnosis

    real :: input(3, 4) = reshape([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.11, 0.12], [3, 4])
    real :: gradient(3, 2) = reshape([0.0, 10., 0.2, 3., 0.4, 1.], [3, 2])
    type(linear2d_layer) :: linear
    real, pointer :: w_ptr(:)
    real, pointer :: b_ptr(:)

    integer :: num_parameters
    real, allocatable :: parameters(:)  ! Remove the fixed size
    real :: expected_parameters(10) = [&
        0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001, 0.100000001,&
        0.109999999, 0.109999999&
    ]
    real :: gradients(10)
    real :: expected_gradients(10) = [&
        1.03999996, 4.09999990, 7.15999985, 1.12400007, 0.240000010, 1.56000006, 2.88000011, 2.86399961,&
        10.1999998, 4.40000010&
    ]
    real :: updated_parameters(10)
    real :: updated_weights(8)
    real :: updated_biases(2)
    real :: expected_weights(8) = [&
        0.203999996, 0.509999990, 0.816000044, 0.212400019, 0.124000005, 0.256000012, 0.388000011, 0.386399955&
    ]
    real :: expected_biases(2) = [1.13000000, 0.550000012]

    integer :: i
    real, parameter :: tolerance = 0.

    linear = linear2d_layer(out_features=2)
    call linear % init([3, 4])
    linear % weights = 0.1
    linear % biases = 0.11
    call linear % forward(input)
    call linear % backward(input, gradient)
    num_parameters = linear % get_num_params()

    test_diagnosis = (num_parameters .equalsExpected. 10) // " (number of parameters)"

    call linear % get_params_ptr(w_ptr, b_ptr)  ! Change this_layer to linear
    allocate(parameters(size(w_ptr) + size(b_ptr)))
    parameters(1:size(w_ptr)) = w_ptr
    parameters(size(w_ptr)+1:) = b_ptr
    test_diagnosis = test_diagnosis .also. (.all. (parameters .approximates. expected_parameters .within. tolerance) // " (parameters)")

    gradients =  linear % get_gradients()
    test_diagnosis = test_diagnosis .also. (.all. (gradients .approximates. expected_gradients .within. tolerance) // " (gradients)")

    do i = 1, num_parameters
      updated_parameters(i) = parameters(i) + 0.1 * gradients(i)
    end do

    call linear % get_params_ptr(w_ptr, b_ptr)  ! Change this_layer to linear
    w_ptr = updated_parameters(1:size(w_ptr))
    b_ptr = updated_parameters(size(w_ptr)+1:)
    updated_weights = reshape(linear % weights, shape(expected_weights))
    test_diagnosis = test_diagnosis .also. (.all. (updated_weights .approximates. expected_weights .within. tolerance) // " (updated weights)")

    updated_biases = linear % biases
    test_diagnosis = test_diagnosis .also. (.all. (updated_biases .approximates. expected_biases .within. tolerance) // " (updated biases)")

  end function

end module linear_2d_layer_test_m