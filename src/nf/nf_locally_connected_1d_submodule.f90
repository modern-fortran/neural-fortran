submodule(nf_locally_connected_1d_layer) nf_locally_connected_1d_layer_submodule

  use nf_activation, only: activation_function
  use nf_random, only: random_normal
  implicit none

contains

  !=====================================================================
  ! Constructor: allocate and initialize a locally connected 1D layer.
  !=====================================================================
  module function locally_connected_1d_layer_cons(filters, kernel_size, activation) result(res)
    integer, intent(in)                   :: filters
    integer, intent(in)                   :: kernel_size
    class(activation_function), intent(in):: activation
    type(locally_connected_1d_layer)       :: res

    res % kernel_size     = kernel_size
    res % filters         = filters
    res % activation_name = activation % get_name()
    allocate(res % activation, source=activation)
  end function locally_connected_1d_layer_cons

  !=====================================================================
  ! Initialize the layer.
  !
  ! Here we assume the input shape is an integer array of length 2:
  !   input_shape(1): number of channels,
  !   input_shape(2): length of the 1D input.
  !
  ! The output length is computed as:
  !   output_length = input_length - kernel_size + 1
  !
  ! The kernel weights are unshared so that each output position gets
  ! its own set of weights. Their shape becomes:
  !   (filters, output_length, channels, kernel_size)
  !
  ! The biases are similarly unshared and allocated with shape:
  !   (filters, output_length)
  !=====================================================================
  module subroutine init(self, input_shape)
    class(locally_connected_1d_layer), intent(in out):: self
    integer, intent(in)                     :: input_shape(:)

    ! Input shape: channels x input_length.
    self % channels     = input_shape(1)
    self % input_length = input_shape(2)
    self % output_length = self % input_length - self % kernel_size + 1

    ! Allocate the output array: shape (filters, output_length)
    allocate(self % output(self % filters, self % output_length))
    self % output = 0

    ! Allocate the kernel.
    ! Kernel shape: (filters, output_length, channels, kernel_size)
    allocate(self % kernel(self % filters, self % output_length, self % channels, self % kernel_size))
    call random_normal(self % kernel)
    self % kernel = self % kernel / self % kernel_size

    ! Allocate the biases: shape (filters, output_length)
    allocate(self % biases(self % filters, self % output_length))
    self % biases = 0

    ! Allocate the pre-activation array, z, with the same shape as output.
    allocate(self % z, mold=self % output)
    self % z = 0

    ! Allocate the gradient for the input.
    allocate(self % gradient(self % channels, self % input_length))
    self % gradient = 0

    ! Allocate the gradients for the kernel and biases.
    allocate(self % dw, mold=self % kernel)
    self % dw = 0

    allocate(self % db, mold=self % biases)
    self % db = 0

  end subroutine init

  !=====================================================================
  ! Forward pass:
  !   For each output position, extract the corresponding patch from
  !   the input (of shape channels x kernel_size), compute the weighted
  !   sum (using the unshared weights for that position), add the bias,
  !   and then apply the activation function.
  !
  ! Input: real array of shape (channels, input_length)
  ! Output: stored in self%output (shape: filters x output_length)
  !=====================================================================
  pure module subroutine forward(self, input)
    class(locally_connected_1d_layer), intent(in out) :: self
    real, intent(in)                                :: input(:,:)
    integer                                         :: pos, n

    ! For each output position, the input patch is:
    !   input(:, pos:pos+kernel_size-1)
    do concurrent (pos = 1:self % output_length)
      do concurrent (n = 1:self % filters)
        self % z(n, pos) = sum( self % kernel(n, pos, :, :) * &
                                input(:, pos:pos+self % kernel_size-1) ) + self % biases(n, pos)
      end do
    end do

    ! Apply the activation function.
    self % output = self % activation % eval(self % z)

  end subroutine forward

  !=====================================================================
  ! Backward pass:
  !   Given the gradient with respect to the output (dL/dy), this
  !   routine computes the gradients with respect to the pre-activation,
  !   the unshared kernel weights, the biases, and the input.
  !
  ! Here, gradient (dL/dy) has shape (filters, output_length).
  !=====================================================================
  pure module subroutine backward(self, input, gradient)
    class(locally_connected_1d_layer), intent(in out) :: self
    real, intent(in)                                :: input(:,:)
    real, intent(in)                                :: gradient(:,:)
    integer                                         :: pos, n
    real, allocatable                             :: gdz(:,:)  ! (filters, output_length)

    ! Allocate a temporary array for the derivative of z.
    allocate(gdz(self % filters, self % output_length))
    ! gdz = dL/dy * activation'(z)
    gdz = gradient * self % activation % eval_prime(self % z)

    ! Update bias gradients. (Each bias is specific to an output position.)
    self % db = self % db + gdz

    ! Reset the gradients for the kernel and input.
    self % dw = 0
    self % gradient = 0

    ! For each output position and filter, compute:
    !  - dL/dw for the weights at that output position, and
    !  - the contribution to dL/dx for the corresponding input patch.
    do concurrent (pos = 1:self % output_length)
      do concurrent (n = 1:self % filters)
        ! The patch from the input corresponding to output position "pos"
        ! is input(:, pos:pos+kernel_size-1).
        self % dw(n, pos, :, :) = self % dw(n, pos, :, :) + &
             input(:, pos:pos+self % kernel_size-1) * gdz(n, pos)

        ! Each such output position contributes to the gradient of the input.
        self % gradient(:, pos:pos+self % kernel_size-1) = &
             self % gradient(:, pos:pos+self % kernel_size-1) + &
             gdz(n, pos) * self % kernel(n, pos, :, :)
      end do
    end do

    deallocate(gdz)

  end subroutine backward

  !=====================================================================
  ! Return the total number of parameters.
  !
  ! For the locally connected layer this equals:
  !   number of elements in the kernel + number of elements in the biases
  !=====================================================================
  pure module function get_num_params(self) result(num_params)
    class(locally_connected_1d_layer), intent(in) :: self
    integer                                      :: num_params

    num_params = product(shape(self % kernel)) + product(shape(self % biases))
  end function get_num_params

  !=====================================================================
  ! Return a flattened array containing all parameters.
  !
  ! The parameters are taken in order: first all kernel weights, then
  ! all biases.
  !=====================================================================
  module function get_params(self) result(params)
    class(locally_connected_1d_layer), intent(in), target :: self
    real, allocatable                                      :: params(:)
    real, pointer                                          :: w_(:) => null()

    w_(1:size(self % kernel)) => self % kernel
    params = [ w_, reshape(self % biases, [product(shape(self % biases))]) ]
  end function get_params

  !=====================================================================
  ! Return a flattened array containing all gradients.
  !
  ! The gradients are taken in order: first all gradients for the kernel,
  ! then the gradients for the biases.
  !=====================================================================
  module function get_gradients(self) result(gradients)
    class(locally_connected_1d_layer), intent(in), target :: self
    real, allocatable                                      :: gradients(:)
    real, pointer                                          :: dw_(:) => null()

    dw_(1:size(self % dw)) => self % dw
    gradients = [ dw_, reshape(self % db, [product(shape(self % db))]) ]
  end function get_gradients

  !=====================================================================
  ! Set the parameters of the layer from a flattened vector.
  !
  ! The parameters vector is assumed to have the same number of elements
  ! as returned by get_num_params.
  !=====================================================================
  module subroutine set_params(self, params)
    class(locally_connected_1d_layer), intent(inout):: self
    real, intent(in)                                :: params(:)
    integer                                         :: num_kernel, num_bias, offset

    num_kernel = product(shape(self % kernel))
    num_bias   = product(shape(self % biases))
    if (size(params) /= num_kernel + num_bias) then
      error stop 'locally connected 1D layer % set_params: Number of parameters does not match'
    end if

    self % kernel = reshape( params(1:num_kernel), shape(self % kernel) )
    offset = num_kernel
    self % biases = reshape( params(offset+1:offset+num_bias), shape(self % biases) )
  end subroutine set_params

end submodule nf_locally_connected_1d_layer_submodule
