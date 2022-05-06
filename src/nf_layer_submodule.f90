submodule(nf_layer) nf_layer_submodule

  use nf_conv2d_layer, only: conv2d_layer
  use nf_dense_layer, only: dense_layer
  use nf_input1d_layer, only: input1d_layer
  use nf_input3d_layer, only: input3d_layer

  implicit none

contains

  pure module subroutine backward(self, previous, gradient)
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: previous
    real, intent(in) :: gradient(:)

    ! Backward pass currently implemented only for dense layers
    select type(this_layer => self % p); type is(dense_layer)

    ! Previous layer is the input layer to this layer.
    ! For a backward pass on a dense layer, we must accept either an input layer
    ! or another dense layer as input.
    select type(prev_layer => previous % p)

      type is(input1d_layer)
        call this_layer % backward(prev_layer % output, gradient)
      type is(dense_layer)
        call this_layer % backward(prev_layer % output, gradient)

    end select
    end select

  end subroutine backward


  pure module subroutine forward(self, input)
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: input

    select type(this_layer => self % p)

      ! Only dense layer is supported for now
      type is(dense_layer)

        ! Input layers permitted: input1d, dense
        select type(prev_layer => input % p)
          type is(input1d_layer)
            call this_layer % forward(prev_layer % output)
          type is(dense_layer)
            call this_layer % forward(prev_layer % output)
        end select

    end select

  end subroutine forward


  pure module subroutine get_output(self, output)
    class(layer), intent(in) :: self
    real, allocatable, intent(out) :: output(:)

    select type(this_layer => self % p)

      type is(input1d_layer)
        allocate(output, source=this_layer % output)
      type is(dense_layer)
        allocate(output, source=this_layer % output)

    end select

  end subroutine get_output


  impure elemental module subroutine init(self, input)
    class(layer), intent(in out) :: self
    class(layer), intent(in) :: input

    if (self % initialized) &
      error stop self % name // ' layer is already initialized.'

    select type(this_layer => self % p); class default
      call this_layer % init(input % layer_shape)
    end select

    ! The shape of a conv2d layer is not known until we receive an input layer.
    select type(this_layer => self % p); type is(conv2d_layer)
      self % layer_shape = shape(this_layer % output)
    end select

    self % input_layer_shape = input % layer_shape 
    self % initialized = .true.

  end subroutine init


  impure elemental module subroutine print_info(self)
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
    class(layer), intent(in out) :: self
    real, intent(in) :: learning_rate

    select type(this_layer => self % p); type is(dense_layer)
      call this_layer % update(learning_rate)
    end select

  end subroutine update

end submodule nf_layer_submodule
