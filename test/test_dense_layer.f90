program test_dense_layer
  use nf, only: dense, layer, relu
  use tuff, only: test, test_result
  implicit none
  type(layer) :: layer1, layer2, layer3
  type(test_result) :: tests

  layer1 = dense(10)
  layer2 = dense(10, activation=relu())
  layer3 = dense(20)
  call layer3 % init(layer1)

  tests = test("test_dense_layer", [ &
    test("layer name is set", layer1 % name == 'dense'), & 
    test("layer shape is correct", all(layer1 % layer_shape == [10])), &
    test("layer is initialized", layer3 % initialized), &
    test("layer's default activation is sigmoid", layer1 % activation == 'sigmoid'), &
    test("user set activation works", layer2 % activation == 'relu'), &
    test("layer initialized after init", layer3 % initialized), &
    test("layer input shape is set after init", all(layer3 % input_layer_shape == [10])) &
  ])  

end program test_dense_layer
