submodule(nf_layer_constructors) nf_layer_constructors_submodule

  use nf_layer, only: layer
  use nf_avgpool1d_layer, only: avgpool1d_layer
  use nf_avgpool2d_layer, only: avgpool2d_layer
  use nf_conv1d_layer, only: conv1d_layer
  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_dropout_layer, only: dropout_layer
  use nf_flatten_layer, only: flatten_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input2d_layer, only: input2d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_locally_connected2d_layer, only: locally_connected2d_layer
  use nf_maxpool1d_layer, only: maxpool1d_layer
  use nf_maxpool2d_layer, only: maxpool2d_layer
  use nf_reshape2d_layer, only: reshape2d_layer
  use nf_reshape3d_layer, only: reshape3d_layer
  use nf_linear2d_layer, only: linear2d_layer
  use nf_self_attention_layer, only: self_attention_layer
  use nf_embedding_layer, only: embedding_layer
  use nf_layernorm_layer, only: layernorm_layer
  use nf_activation, only: relu, sigmoid

  implicit none

contains

  module function conv1d(filters, kernel_width, activation, stride) result(res)
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_width
    class(activation_function), intent(in), optional :: activation
    integer, intent(in), optional :: stride
    type(layer) :: res

    integer :: stride_tmp
    class(activation_function), allocatable :: activation_tmp

    res % name = 'conv1d'

    if (present(activation)) then
      allocate(activation_tmp, source=activation)
    else
      allocate(activation_tmp, source=relu())
    end if

    res % activation = activation_tmp % get_name()

    if (present(stride)) then
      stride_tmp = stride
    else
      stride_tmp = 1
    endif

    if (stride_tmp < 1) &
      error stop 'stride must be >= 1 in a conv1d layer'

    allocate( &
      res % p, &
      source=conv1d_layer(filters, kernel_width, activation_tmp, stride_tmp) &
    )

  end function conv1d

  module function conv2d(filters, kernel_width, kernel_height, activation, stride) result(res)
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_width
    integer, intent(in) :: kernel_height
    class(activation_function), intent(in), optional :: activation
    integer, intent(in), optional :: stride(:)
    type(layer) :: res

    integer, allocatable :: stride_tmp(:)
    class(activation_function), allocatable :: activation_tmp

    ! Enforce kernel_width == kernel_height for now;
    ! If non-square kernels show to be desired, we'll relax this constraint
    ! and refactor conv2d_layer to work with non-square kernels.
    if (kernel_width /= kernel_height) &
      error stop 'kernel_width must equal kernel_height in a conv2d layer'

    res % name = 'conv2d'

    if (present(activation)) then
      allocate(activation_tmp, source=activation)
    else
      allocate(activation_tmp, source=relu())
    end if

    res % activation = activation_tmp % get_name()

    if (present(stride)) then
      stride_tmp = stride
    else
      stride_tmp = [1, 1]
    endif

    if (size(stride_tmp) /= 2 ) &
      error stop 'size of stride must be equal to 2 in a conv2d layer'

    if (stride_tmp(1) < 1 .or. stride_tmp(2) < 1) &
      error stop 'stride must be >= 1 in a conv2d layer'

    allocate( &
      res % p, &
      source=conv2d_layer(filters, kernel_width, activation_tmp, stride_tmp) &
    )

  end function conv2d

  module function locally_connected2d(filters, kernel_size, activation, stride) result(res)
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in), optional :: activation
    integer, intent(in), optional :: stride
    type(layer) :: res

    integer :: stride_tmp
    class(activation_function), allocatable :: activation_tmp

    res % name = 'locally_connected2d'

    if (present(activation)) then
      allocate(activation_tmp, source=activation)
    else
      allocate(activation_tmp, source=relu())
    end if

    res % activation = activation_tmp % get_name()

    if (present(stride)) then
      stride_tmp = stride
    else
      stride_tmp = 1
    endif

    if (stride_tmp < 1) &
      error stop 'stride must be >= 1 in a conv1d layer'

    allocate( &
      res % p, &
      source=locally_connected2d_layer(filters, kernel_size, activation_tmp, stride_tmp) &
    )

  end function locally_connected2d


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


  module function avgpool1d(pool_width, stride) result(res)
    integer, intent(in) :: pool_width
    integer, intent(in) :: stride
    type(layer) :: res

    if (pool_width < 2) &
      error stop 'pool_width must be >= 2 in a avgpool1d layer'

    if (stride < 1) &
      error stop 'stride must be >= 1 in a avgpool1d layer'

    res % name = 'avgpool1d'

    allocate( &
      res % p, &
      source=avgpool1d_layer(pool_width, stride) &
    )

  end function avgpool1d

  module function avgpool2d(pool_width, pool_height, stride) result(res)
    integer, intent(in) :: pool_width
    integer, intent(in) :: pool_height
    integer, intent(in) :: stride
    type(layer) :: res

    if (pool_width < 2) &
      error stop 'pool_width must be >= 2 in a avgpool2d layer'

    if (pool_height < 2) &
      error stop 'pool_height must be >= 2 in a avgpool2d layer'

    if (stride < 1) &
      error stop 'stride must be >= 1 in a avgpool2d layer'

    res % name = 'avgpool2d'

    allocate( &
      res % p, &
      source=avgpool2d_layer(pool_width, pool_height, stride) &
    )

  end function avgpool2d

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

  module function maxpool1d(pool_width, stride) result(res)
    integer, intent(in) :: pool_width
    integer, intent(in) :: stride
    type(layer) :: res

    if (pool_width < 2) &
      error stop 'pool_width must be >= 2 in a maxpool1d layer'

    if (stride < 1) &
      error stop 'stride must be >= 1 in a maxpool1d layer'

    res % name = 'maxpool1d'

    allocate( &
      res % p, &
      source=maxpool1d_layer(pool_width, stride) &
    )

  end function maxpool1d

  module function maxpool2d(pool_width, pool_height, stride) result(res)
    integer, intent(in) :: pool_width
    integer, intent(in) :: pool_height
    integer, intent(in) :: stride
    type(layer) :: res

    if (pool_width < 2) &
      error stop 'pool_width must be >= 2 in a maxpool2d layer'

    ! Enforce pool_width == pool_height for now;
    ! If non-square poolings show to be desired, we'll relax this constraint
    ! and refactor maxpool2d_layer to work with non-square kernels.
    if (pool_width /= pool_height) &
      error stop 'pool_width must equal pool_height in a maxpool2d layer'

    if (stride < 1) &
      error stop 'stride must be >= 1 in a maxpool2d layer'

    res % name = 'maxpool2d'

    allocate( &
      res % p, &
      source=maxpool2d_layer(pool_width, stride) &
    )

  end function maxpool2d

  module function reshape2d(dim1, dim2) result(res)
    integer, intent(in) :: dim1, dim2
    type(layer) :: res
    res % name = 'reshape2d'
    res % layer_shape = [dim1, dim2]
    allocate(res % p, source=reshape2d_layer(res % layer_shape))
  end function reshape2d


  module function reshape3d(dim1, dim2, dim3) result(res)
    integer, intent(in) :: dim1, dim2, dim3
    type(layer) :: res
    res % name = 'reshape3d'
    res % layer_shape = [dim1, dim2, dim3]
    allocate(res % p, source=reshape3d_layer(res % layer_shape))
  end function reshape3d


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


  module function embedding(sequence_length, vocab_size, model_dimension, positional) result(res)
    integer, intent(in) :: sequence_length, vocab_size, model_dimension
    integer, optional, intent(in) :: positional
    type(layer) :: res
    type(embedding_layer) :: embedding_layer_instance

    embedding_layer_instance = embedding_layer(vocab_size, model_dimension, positional)
    call embedding_layer_instance % init([sequence_length])
    res % name = 'embedding'
    res % layer_shape = [sequence_length, model_dimension]
    res % input_layer_shape = [integer ::]
    allocate(res % p, source=embedding_layer_instance)
    res % initialized = .true.

  end function embedding


  module function layernorm() result(res)
    type(layer) :: res
    res % name = 'layernorm'
    allocate(res % p, source=layernorm_layer())
  end function layernorm

end submodule nf_layer_constructors_submodule
