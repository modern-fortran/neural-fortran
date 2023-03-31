submodule(nf_conv2d_layer) nf_conv2d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: randn

  implicit none

contains

  pure module function conv2d_layer_cons(filters, kernel_size, activation) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in) :: activation
    type(conv2d_layer) :: res

    res % kernel_size = kernel_size
    res % filters = filters
    res % activation_name = activation % get_name()
    allocate( res % activation, source = activation )

  end function conv2d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)
    self % width = input_shape(2) - self % kernel_size + 1
    self % height = input_shape(3) - self % kernel_size + 1

    ! Output of shape filters x width x height
    allocate(self % output(self % filters, self % width, self % height))
    self % output = 0

    ! Kernel of shape filters x channels x width x height
    allocate(self % kernel(self % filters, self % channels, &
                           self % kernel_size, self % kernel_size))

    ! Initialize the kernel with random values with a normal distribution.
    self % kernel = randn(self % filters, self % channels, &
                          self % kernel_size, self % kernel_size) &
                  / self % kernel_size**2 !TODO kernel_width * kernel_height

    allocate(self % biases(self % filters))
    self % biases = 0

    allocate(self % z, mold=self % output)
    self % z = 0

    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3)))
    self % gradient = 0

    allocate(self % dw, mold=self % kernel)
    self % dw = 0

    allocate(self % db, mold=self % biases)
    self % db = 0

  end subroutine init


  pure module subroutine forward(self, input)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    integer :: input_width, input_height, input_channels
    integer :: istart, iend
    integer :: jstart, jend
    integer :: i, j, n
    integer :: iws, iwe, jws, jwe
    integer :: half_window

    ! Input dimensions are channels x width x height
    input_channels = size(input, dim=1)
    input_width = size(input, dim=2)
    input_height = size(input, dim=3)

    ! Half-window is 1 for window size 3; 2 for window size 5; etc.
    half_window = self % kernel_size / 2

    ! Determine the start and end indices for the width and height dimensions
    ! of the input that correspond to the center of each window.
    istart = half_window + 1 ! TODO kernel_width
    jstart = half_window + 1 ! TODO kernel_height
    iend = input_width - istart + 1
    jend = input_height - jstart + 1

    convolution: do concurrent(i = istart:iend, j = jstart:jend)

      ! Start and end indices of the input data on the filter window
      ! iws and jws are also coincidentally the indices of the output matrix
      iws = i - half_window ! TODO kernel_width
      iwe = i + half_window ! TODO kernel_width
      jws = j - half_window ! TODO kernel_height
      jwe = j + half_window ! TODO kernel_height

      ! Compute the inner tensor product, sum(w_ij * x_ij), for each filter.
      do concurrent(n = 1:self % filters)
        self % z(n,iws,jws) = sum(self % kernel(n,:,:,:) * input(:,iws:iwe,jws:jwe))
      end do

      ! Add bias to the inner product.
      self % z(:,iws,jws) = self % z(:,iws,jws) + self % biases

    end do convolution

    ! Activate
    self % output = self % activation % eval(self % z)

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    real :: db(self % filters)
    real :: dw(self % filters, self % channels, self % kernel_size, self % kernel_size)
    real :: gdz(self % filters, size(input, 2), size(input, 3))
    integer :: i, j, k, n
    integer :: input_channels, input_width, input_height
    integer :: istart, iend
    integer :: jstart, jend
    integer :: iws, iwe, jws, jwe
    integer :: half_window

    ! Input dimensions are channels x width x height.
    ! Input frame goes from 1 to input_width and from 1 to input_height.
    input_channels = size(input, dim=1)
    input_width = size(input, dim=2)
    input_height = size(input, dim=3)

    ! Half-window is 1 for window size 3; 2 for window size 5; etc.
    half_window = self % kernel_size / 2

    ! Determine the start and end indices for the width and height dimensions
    ! of the input that correspond to the center of each window.
    istart = half_window + 1 ! TODO kernel_width
    jstart = half_window + 1 ! TODO kernel_height
    iend = input_width - istart + 1
    jend = input_height - jstart + 1

    ! z = w .inner. x + b
    ! gdz = dL/dy * sigma'(z)
    gdz = 0
    gdz(:,istart:iend,jstart:jend) = gradient * self % activation % eval_prime(self % z)

    ! dL/db = sum(dL/dy * sigma'(z))
    do concurrent (n = 1:self % filters)
      db(n) = sum(gdz(n,:,:))
    end do

    dw = 0
    self % gradient = 0
    do concurrent( &
      n = 1:self % filters, &
      k = 1:self % channels, &
      i = istart:iend, &
      j = jstart:jend &
    )
      ! Start and end indices of the input data on the filter window
      iws = i - half_window ! TODO kernel_width
      iwe = i + half_window ! TODO kernel_width
      jws = j - half_window ! TODO kernel_height
      jwe = j + half_window ! TODO kernel_height

      ! dL/dw = sum(dL/dy * sigma'(z) * x)
      dw(n,k,:,:) = dw(n,k,:,:) + input(k,iws:iwe,jws:jwe) * gdz(n,iws:iwe,jws:jwe)

      ! dL/dx = dL/dy * sigma'(z) .inner. w
      self % gradient(k,i,j) = self % gradient(k,i,j) &
        + sum(gdz(n,iws:iwe,jws:jwe) * self % kernel(n,k,:,:))

    end do

    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward


  pure module function get_num_params(self) result(num_params)
    class(conv2d_layer), intent(in) :: self
    integer :: num_params
    num_params = product(shape(self % kernel)) + size(self % biases)
  end function get_num_params


  pure module function get_params(self) result(params)
    class(conv2d_layer), intent(in) :: self
    real, allocatable :: params(:)

    params = [ &
      pack(self % kernel, .true.), &
      pack(self % biases, .true.) &
    ]

  end function get_params


  module subroutine set_params(self, params)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: params(:)

    ! Check that the number of parameters is correct.
    if (size(params) /= self % get_num_params()) then
       error stop 'conv2d % set_params: Number of parameters does not match'
    end if

    ! Reshape the kernel.
    self % kernel = reshape( &
      params(:product(shape(self % kernel))), &
      shape(self % kernel) &
    )

    ! Reshape the biases.
    self % biases = reshape( &
      params(product(shape(self % kernel)) + 1:), &
      [self % filters] &
    )

  end subroutine set_params


  module subroutine update(self, learning_rate)
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: learning_rate

    ! Sum weight and bias gradients across images, if any
    call co_sum(self % dw)
    call co_sum(self % db)

    self % kernel = self % kernel - learning_rate * self % dw
    self % biases = self % biases - learning_rate * self % db
    self % dw = 0
    self % db = 0

  end subroutine update

end submodule nf_conv2d_layer_submodule
