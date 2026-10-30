program test_flatten_layer
  use nf, only: dense, flatten, input, layer, network
  use nf_flatten_layer, only: flatten_layer
  use nf_input2d_layer, only: input2d_layer
  use nf_input3d_layer, only: input3d_layer
  use tuff, only: test, test_result
  implicit none

  type(layer) :: input_layer, layer1, layer2
  type(test_result) :: tests

  layer1 = flatten()
  input_layer = input(1, 2, 2)
  layer2 = flatten()
  call layer2 % init(input_layer)

  tests = test("test_flatten_layer", [ &
    test("layer name is set", layer1 % name == "flatten"), &
    test("layer is not initialized by default", .not. layer1 % initialized), &
    test("layer initializes from 3D input", layer2 % initialized), &
    test("layer shape is correct for 3D input", all(layer2 % layer_shape == [4])), &
    test(initializes_from_2d_input), &
    test(forward_3d_input), &
    test(backward_3d_input), &
    test(forward_2d_input), &
    test(backward_2d_input), &
    test(chains_input3d_to_dense) &
  ])

contains

  type(test_result) function initializes_from_2d_input() result(res)
    type(layer) :: input_layer, test_layer

    res % name = "layer initializes from 2D input"

    input_layer = input(2, 3)
    test_layer = flatten()
    call test_layer % init(input_layer)

    res % ok = test_layer % initialized
    if (.not. res % ok) return

    res % ok = all(test_layer % layer_shape == [6])
  end function initializes_from_2d_input


  type(test_result) function forward_3d_input() result(res)
    type(layer) :: input_layer, test_layer
    real, allocatable :: output(:)

    res % name = "forward propagates 3D input"

    input_layer = input(1, 2, 2)
    test_layer = flatten()
    call test_layer % init(input_layer)

    select type(p => input_layer % p)
      type is (input3d_layer)
        call p % set(reshape(real([1, 2, 3, 4]), [1, 2, 2]))
      class default
        res % ok = .false.
        return
    end select

    call test_layer % forward(input_layer)
    call test_layer % get_output(output)

    res % ok = allocated(output)
    if (.not. res % ok) return

    res % ok = size(output) == 4
    if (.not. res % ok) return

    res % ok = all(output == real([1, 2, 3, 4]))
  end function forward_3d_input


  type(test_result) function backward_3d_input() result(res)
    type(layer) :: input_layer, test_layer

    res % name = "backward propagates 3D gradient"

    input_layer = input(1, 2, 2)
    test_layer = flatten()
    call test_layer % init(input_layer)

    call test_layer % backward(input_layer, real([1, 2, 3, 4]))

    select type(p => test_layer % p)
      type is (flatten_layer)
        res % ok = allocated(p % gradient_3d)
        if (.not. res % ok) return

        res % ok = all(shape(p % gradient_3d) == [1, 2, 2])
        if (.not. res % ok) return

        res % ok = all(p % gradient_3d == reshape(real([1, 2, 3, 4]), [1, 2, 2]))
      class default
        res % ok = .false.
    end select
  end function backward_3d_input


  type(test_result) function forward_2d_input() result(res)
    type(layer) :: input_layer, test_layer
    real, allocatable :: output(:)

    res % name = "forward propagates 2D input"

    input_layer = input(2, 3)
    test_layer = flatten()
    call test_layer % init(input_layer)

    select type(p => input_layer % p)
      type is (input2d_layer)
        call p % set(reshape(real([1, 2, 3, 4, 5, 6]), [2, 3]))
      class default
        res % ok = .false.
        return
    end select

    call test_layer % forward(input_layer)
    call test_layer % get_output(output)

    res % ok = allocated(output)
    if (.not. res % ok) return

    res % ok = size(output) == 6
    if (.not. res % ok) return

    res % ok = all(output == real([1, 2, 3, 4, 5, 6]))
  end function forward_2d_input


  type(test_result) function backward_2d_input() result(res)
    type(layer) :: input_layer, test_layer

    res % name = "backward propagates 2D gradient"

    input_layer = input(2, 3)
    test_layer = flatten()
    call test_layer % init(input_layer)

    call test_layer % backward(input_layer, real([1, 2, 3, 4, 5, 6]))

    select type(p => test_layer % p)
      type is (flatten_layer)
        res % ok = allocated(p % gradient_2d)
        if (.not. res % ok) return

        res % ok = all(shape(p % gradient_2d) == [2, 3])
        if (.not. res % ok) return

        res % ok = all(p % gradient_2d == reshape(real([1, 2, 3, 4, 5, 6]), [2, 3]))
      class default
        res % ok = .false.
    end select
  end function backward_2d_input


  type(test_result) function chains_input3d_to_dense() result(res)
    type(network) :: net

    res % name = "chains input3d to dense"

    net = network([ &
      input(1, 28, 28), &
      flatten(), &
      dense(10) &
    ])

    res % ok = all(net % layers(3) % input_layer_shape == [784])
  end function chains_input3d_to_dense

end program test_flatten_layer
