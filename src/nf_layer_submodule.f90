submodule(nf_layer) nf_layer_submodule

  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_flatten_layer, only: flatten_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input3d_layer, only: input3d_layer
  use nf_maxpool2d_layer, only: maxpool2d_layer

contains

  pure module subroutine backward(self, previous, gradient)
    implicit none
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: previous
    real, intent(in) :: gradient(:)

    ! Backward pass currently implemented only for dense layers
    select type(this_layer => self % p)

      type is(dense_layer)

        ! Upstream layers permitted: input1d, dense, flatten
        select type(prev_layer => previous % p)
          type is(input1d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(dense_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(flatten_layer)
            call this_layer % backward(prev_layer % output, gradient)
        end select

      type is(flatten_layer)

        ! Downstream layers permitted: input3d, conv2d, maxpool2d
        select type(prev_layer => previous % p)
          type is(input3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(conv2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(maxpool2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
        end select

    end select

  end subroutine backward


  pure module subroutine forward(self, input)
    implicit none
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: input

    select type(this_layer => self % p)

      type is(dense_layer)

        ! Upstream layers permitted: input1d, dense, flatten
        select type(prev_layer => input % p)
          type is(input1d_layer)
            call this_layer % forward(prev_layer % output)
          type is(dense_layer)
            call this_layer % forward(prev_layer % output)
          type is(flatten_layer)
            call this_layer % forward(prev_layer % output)
        end select

      type is(conv2d_layer)

        ! Upstream layers permitted: input3d, conv2d, maxpool2d
        select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
        end select

      type is(maxpool2d_layer)

        ! Upstream layers permitted: input3d, conv2d, maxpool2d
        select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
        end select

      type is(flatten_layer)

        ! Upstream layers permitted: input3d, conv2d, maxpool2d
        select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
        end select

    end select

  end subroutine forward


  pure module subroutine get_output_1d(self, output)
    implicit none
    class(layer), intent(in) :: self
    real, allocatable, intent(out) :: output(:)

    select type(this_layer => self % p)

      type is(input1d_layer)
        allocate(output, source=this_layer % output)
      type is(dense_layer)
        allocate(output, source=this_layer % output)
      type is(flatten_layer)
        allocate(output, source=this_layer % output)
      class default
        error stop '1-d output can only be read from an input1d, dense, or flatten layer.'

    end select

  end subroutine get_output_1d


  pure module subroutine get_output_3d(self, output)
    implicit none
    class(layer), intent(in) :: self
    real, allocatable, intent(out) :: output(:,:,:)

    select type(this_layer => self % p)

      type is(input3d_layer)
        allocate(output, source=this_layer % output)
      type is(conv2d_layer)
        allocate(output, source=this_layer % output)
      type is(maxpool2d_layer)
        allocate(output, source=this_layer % output)
      class default
        error stop '3-d output can only be read from an input3d, conv2d, or maxpool2d layer.'

    end select

  end subroutine get_output_3d


  impure elemental module subroutine init(self, input)
    implicit none
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: input

    if (self % initialized) &
      error stop self % name // ' layer is already initialized.'

    select type(this_layer => self % p); class default
      call this_layer % init(input % layer_shape)
    end select

    ! The shape of conv2d, maxpool2d, or flatten layers is not known
    ! until we receive an input layer.
    select type(this_layer => self % p)
      type is(conv2d_layer)
        self % layer_shape = shape(this_layer % output)
      type is(maxpool2d_layer)
        self % layer_shape = shape(this_layer % output)
      type is(flatten_layer)
        self % layer_shape = shape(this_layer % output)
    end select

    self % input_layer_shape = input % layer_shape 
    self % initialized = .true.

  end subroutine init


  impure elemental module subroutine print_info(self)
    implicit none
    class(layer), intent(in) :: self
    print '("Layer: ", a)', self % name
    print '(60("-"))'
    if (.not. self % name == 'input') &
      print '("Input shape: ", *(i0, 1x))', self % input_layer_shape
    print '("Output shape: ", *(i0, 1x))', self % layer_shape
    if (.not. self % name == 'input') &
      print '("Activation: ", a)', self % activation
    print *
  end subroutine print_info


  impure elemental module subroutine update(self, learning_rate)
    implicit none
    class(layer), intent(in out) :: self
    real, intent(in) :: learning_rate

    select type(this_layer => self % p); type is(dense_layer)
      call this_layer % update(learning_rate)
    end select

  end subroutine update

end submodule nf_layer_submodule
