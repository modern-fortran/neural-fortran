submodule(nf_layer_constructors) nf_layer_constructors_submodule

  use nf_layer, only: layer
  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_dropout_layer, only: dropout_layer
  use nf_flatten_layer, only: flatten_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input2d_layer, only: input2d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_maxpool2d_layer, only: maxpool2d_layer
  use nf_reshape_layer, only: reshape3d_layer
  use nf_linear2d_layer, only: linear2d_layer
  use nf_self_attention_layer, only: self_attention_layer
  use nf_layernorm_layer, only: layernorm_layer
  use nf_activation, only: activation_function, relu, sigmoid

  implicit none

contains

  module function conv2d(filters, kernel_size, activation) result(res)
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in), optional :: activation
    type(layer) :: res

    class(activation_function), allocatable :: activation_tmp

    res % name = 'conv2d'

    if (present(activation)) then
      allocate(activation_tmp, source=activation)
    else
      allocate(activation_tmp, source=relu())
    end if

    res % activation = activation_tmp % get_name()

    allocate( &
      res % p, &
      source=conv2d_layer(filters, kernel_size, activation_tmp) &
    )

  end function conv2d


  module function dense(layer_size, activation) result(res)
    integer, intent(in) :: layer_size
    class(activation_function), intent(in), optional :: activation
    type(layer) :: res

    class(activation_function), allocatable :: activation_tmp

    res % name = 'dense'
    res % layer_shape = [layer_size]

    if (present(activation)) then
      allocate(activation_tmp, source=activation)
    else
      allocate(activation_tmp, source=sigmoid())
    end if

    res % activation = activation_tmp % get_name()

    allocate(res % p, source=dense_layer(layer_size, activation_tmp))

  end function dense


  module function dropout(rate) result(res)
    real, intent(in) :: rate
    type(layer) :: res
    if (rate < 0 .or. rate > 1) &
      error stop 'rate must be between 0 and 1 in a dropout layer'
    res % name = 'dropout'
    allocate(res % p, source=dropout_layer(rate))
  end function dropout


  module function flatten() result(res)
    type(layer) :: res
    res % name = 'flatten'
    allocate(res % p, source=flatten_layer())
  end function flatten


  module function input1d(layer_size) result(res)
    integer, intent(in) :: layer_size
    type(layer) :: res
    res % name = 'input'
    res % layer_shape = [layer_size]
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=input1d_layer(layer_size))
    res % initialized = .true.
  end function input1d


  module function input2d(dim1, dim2) result(res)
    integer, intent(in) :: dim1, dim2
    type(layer) :: res
    res % name = 'input'
    res % layer_shape = [dim1, dim2]
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=input2d_layer([dim1, dim2]))
    res % initialized = .true.
  end function input2d


  module function input3d(dim1, dim2, dim3) result(res)
    integer, intent(in) :: dim1, dim2, dim3
    type(layer) :: res
    res % name = 'input'
    res % layer_shape = [dim1, dim2, dim3]
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=input3d_layer([dim1, dim2, dim3]))
    res % initialized = .true.
  end function input3d


  module function maxpool2d(pool_size, stride) result(res)
    integer, intent(in) :: pool_size
    integer, intent(in), optional :: stride
    integer :: stride_
    type(layer) :: res

    if (pool_size < 2) &
      error stop 'pool_size must be >= 2 in a maxpool2d layer'

    ! Stride defaults to pool_size if not provided
    if (present(stride)) then
      stride_ = stride
    else
      stride_ = pool_size
    end if

    if (stride_ < 1) &
      error stop 'stride must be >= 1 in a maxpool2d layer'

    res % name = 'maxpool2d'

    allocate( &
      res % p, &
      source=maxpool2d_layer(pool_size, stride_) &
    )

  end function maxpool2d


  module function reshape(output_shape) result(res)
    integer, intent(in) :: output_shape(:)
    type(layer) :: res

    res % name = 'reshape'
    res % layer_shape = output_shape

    if (size(output_shape) == 3) then
      allocate(res % p, source=reshape3d_layer(output_shape))
    else
      error stop 'size(output_shape) of the reshape layer must == 3'
    end if

  end function reshape


  module function linear2d(out_features) result(res)
    integer, intent(in) :: out_features
    type(layer) :: res

    res % name = 'linear2d'
    allocate(res % p, source=linear2d_layer(out_features))

  end function linear2d

  module function self_attention(num_heads) result(res)
    integer, intent(in) :: num_heads
    type(layer) :: res

    res % name = 'self_attention'
    allocate(res % p, source=self_attention_layer(num_heads))
  end function self_attention

  module function layernorm() result(res)
    type(layer) :: res

    res % name = 'layernorm'
    allocate(res % p, source=layernorm_layer())
  end function layernorm

end submodule nf_layer_constructors_submodule
