submodule(nf_conv3d_layer) nf_conv3d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: random_normal

  implicit none

contains

  module function conv3d_layer_cons(filters, kernel_size, activation, stride) result(res)
    implicit none
    integer, intent(in) :: filters
    integer, intent(in) :: kernel_size
    class(activation_function), intent(in) :: activation
    integer, intent(in) :: stride(:)
    type(conv3d_layer) :: res

    res % kernel_size = kernel_size
    res % filters = filters
    res % activation_name = activation % get_name()
    res % stride = stride
    allocate( res % activation, source = activation )

  end function conv3d_layer_cons


  module subroutine init(self, input_shape)
    implicit none
    class(conv3d_layer), intent(in out) :: self
    integer, intent(in) :: input_shape(:)

    self % channels = input_shape(1)

    self % width = (input_shape(2) - self % kernel_size) / self % stride(1) + 1
    if (mod(input_shape(2) - self % kernel_size , self % stride(1)) /= 0) self % width = self % width + 1

    self % height = (input_shape(3) - self % kernel_size) / self % stride(2) + 1
    if (mod(input_shape(3) - self % kernel_size , self % stride(2)) /= 0) self % height = self % height + 1

    self % depth = (input_shape(4) - self % kernel_size) / self % stride(3) + 1
    if (mod(input_shape(4) - self % kernel_size , self % stride(3)) /= 0) self % depth = self % depth + 1

    ! Output of shape filters x width x height x depth
    allocate(self % output(self % filters, self % width, self % height, self % depth))
    self % output = 0

    ! Kernel of shape filters x channels x width x height x depth
    allocate(self % kernel(self % filters, self % channels, &
                           self % kernel_size, self % kernel_size, self % kernel_size))

    ! Initialize the kernel with random values with a normal distribution.
    call random_normal(self % kernel)
    self % kernel = self % kernel / self % kernel_size**3

    allocate(self % biases(self % filters))
    self % biases = 0

    allocate(self % z, mold=self % output)
    self % z = 0

    allocate(self % gradient(input_shape(1), input_shape(2), input_shape(3), input_shape(4)))
    self % gradient = 0

    allocate(self % dw, mold=self % kernel)
    self % dw = 0

    allocate(self % db, mold=self % biases)
    self % db = 0

  end subroutine init


  pure module subroutine forward(self, input)
    class(conv3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    integer :: input_width, input_height, input_depth
    integer :: i, j, k, n
    integer :: iws, iwe, jws, jwe, kws, kwe

    ! Input dimensions are channels x width x height x depth
    input_width = size(input, dim=2)
    input_height = size(input, dim=3)
    input_depth = size(input, dim=4)

    convolution: do concurrent(i = 1:self % width, j = 1:self % height, k = 1:self % depth)

      ! Start and end indices of the input data on the filter window
      iws = self % stride(1) * (i - 1) + 1
      iwe = min(iws + self % kernel_size - 1, input_width)

      jws = self % stride(2) * (j - 1) + 1
      jwe = min(jws + self % kernel_size - 1, input_height)

      kws = self % stride(3) * (k - 1) + 1
      kwe = min(kws + self % kernel_size - 1, input_depth)

      ! Compute the inner tensor product, sum(w_ijk * x_ijk), for each filter.
      do concurrent(n = 1:self % filters)
        self % z(n,i,j,k) = sum(self % kernel(n,:,1:iwe-iws+1,1:jwe-jws+1,1:kwe-kws+1) &
          * input(:,iws:iwe,jws:jwe,kws:kwe))
      end do

      ! Add bias to the inner product.
      self % z(:,i,j,k) = self % z(:,i,j,k) + self % biases

    end do convolution

    ! Activate
    self % output = self % activation % eval(self % z)

  end subroutine forward


  pure module subroutine backward(self, input, gradient)
    class(conv3d_layer), intent(in out) :: self
    real, intent(in) :: input(:,:,:,:)
    real, intent(in) :: gradient(:,:,:,:)
    real :: db(self % filters)
    real :: dw(self % filters, self % channels, self % kernel_size, self % kernel_size, self % kernel_size)
    real :: gdz(self % filters, self % width, self % height, self % depth)
    integer :: i, j, k, n, c
    integer :: input_width, input_height, input_depth
    integer :: iws, iwe, jws, jwe, kws, kwe

    ! Input dimensions are channels x width x height x depth.
    input_width = size(input, dim=2)
    input_height = size(input, dim=3)
    input_depth = size(input, dim=4)

    ! z = w .inner. x + b
    ! gdz = dL/dy * sigma'(z)
    gdz = gradient * self % activation % eval_prime(self % z)

    ! dL/db = sum(dL/dy * sigma'(z))
    do concurrent (n = 1:self % filters)
      db(n) = sum(gdz(n,:,:,:))
    end do

    dw = 0
    self % gradient = 0
    do n = 1, self % filters
      do i = 1, self % width
        do j = 1, self % height
          do k = 1, self % depth
            ! Start and end indices of the input data on the filter window
            iws = self % stride(1) * (i - 1) + 1
            iwe = min(iws + self % kernel_size - 1, input_width)

            jws = self % stride(2) * (j - 1) + 1
            jwe = min(jws + self % kernel_size - 1, input_height)

            kws = self % stride(3) * (k - 1) + 1
            kwe = min(kws + self % kernel_size - 1, input_depth)

            do c = 1, self % channels
              ! dL/dw = sum(gdz * x)
              dw(n,c,1:iwe-iws+1,1:jwe-jws+1,1:kwe-kws+1) = &
                dw(n,c,1:iwe-iws+1,1:jwe-jws+1,1:kwe-kws+1) &
                + input(c,iws:iwe,jws:jwe,kws:kwe) * gdz(n,i,j,k)

              ! dL/dx = sum(gdz * w)
              self % gradient(c,iws:iwe,jws:jwe,kws:kwe) = &
                self % gradient(c,iws:iwe,jws:jwe,kws:kwe) &
                + gdz(n,i,j,k) * self % kernel(n,c,1:iwe-iws+1,1:jwe-jws+1,1:kwe-kws+1)
            end do
          end do
        end do
      end do
    end do

    self % dw = self % dw + dw
    self % db = self % db + db

  end subroutine backward


  pure module function get_num_params(self) result(num_params)
    class(conv3d_layer), intent(in) :: self
    integer :: num_params
    num_params = product(shape(self % kernel)) + size(self % biases)
  end function get_num_params


  module subroutine get_params_ptr(self, w_ptr, b_ptr)
    class(conv3d_layer), intent(in), target :: self
    real, pointer, intent(out) :: w_ptr(:)
    real, pointer, intent(out) :: b_ptr(:)
    w_ptr(1:size(self % kernel)) => self % kernel
    b_ptr => self % biases
  end subroutine get_params_ptr


  module subroutine get_gradients_ptr(self, dw_ptr, db_ptr)
    class(conv3d_layer), intent(in), target :: self
    real, pointer, intent(out) :: dw_ptr(:)
    real, pointer, intent(out) :: db_ptr(:)
    dw_ptr(1:size(self % dw)) => self % dw
    db_ptr => self % db
  end subroutine get_gradients_ptr

end submodule nf_conv3d_layer_submodule
