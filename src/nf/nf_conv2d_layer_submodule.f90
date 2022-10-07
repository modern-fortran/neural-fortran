submodule(nf_conv2d_layer) nf_conv2d_layer_submodule

  use nf_random, only: randn

  implicit none

contains

  pure module function conv2d_layer_cons(filters, kernel_size, activation) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    character(*), intent(in) :: activation
    type(conv2d_layer) :: res
    res % kernel_size = kernel_size
    res % filters = filters
    call res % set_activation(activation)
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

      ! Activate
      self % output(:,iws,jws) = self % activation(self % z(:,iws,jws))

    end do convolution

  end subroutine forward


  module subroutine backward(self, input, gradient)
    implicit none
    class(conv2d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:)
    real, intent(in) :: gradient(:,:,:)
    real :: db(self % filters)
    real :: dw(self % filters, self % channels, self % kernel_size, self % kernel_size)
    integer :: i, j, k, n
    integer :: input_channels, input_width, input_height
    integer :: istart, iend
    integer :: jstart, jend
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

    do concurrent (n = 1:self % filters)
      db(n) = sum(gradient(n,:,:) * self % activation_prime(self % z(n,:,:)))
    end do

    dw = 0
    convolution: do concurrent(i = istart:iend, j = jstart:jend)
      do k = 1, self % channels
        do n = 1, self % filters
          ! Start and end indices of the input data on the filter window
          ! iws and jws are also coincidentally the indices of the output matrix
          iws = i - half_window ! TODO kernel_width
          iwe = i + half_window ! TODO kernel_width
          jws = j - half_window ! TODO kernel_height
          jwe = j + half_window ! TODO kernel_height
          dw(n,k,:,:) = dw(n,k,:,:) &
            + input(k,iws:iwe,jws:jwe) &
            * gradient(n,iws,jws) &
            * self % activation_prime(self % z(n,iws,jws))
        end do
      end do
    end do convolution

    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward

end submodule nf_conv2d_layer_submodule
