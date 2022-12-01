submodule(nf_layer) nf_layer_submodule

   use nf_conv2d_layer, only: conv2d_layer
   use nf_dense_layer, only: dense_layer
   use nf_flatten_layer, only: flatten_layer
   use nf_input1d_layer, only: input1d_layer
   use nf_input3d_layer, only: input3d_layer
   use nf_maxpool2d_layer, only: maxpool2d_layer
   use nf_reshape_layer, only: reshape3d_layer

contains

   pure module subroutine backward_1d(self, previous, gradient)
      implicit none
      class(layer), intent(in out) :: self
      class(layer), intent(in) :: previous
      real, intent(in) :: gradient(:)

      ! Backward pass from a 1-d layer downstream currently implemented
      ! only for dense and flatten layers
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

         ! Upstream layers permitted: input3d, conv2d, maxpool2d
         select type(prev_layer => previous % p)
          type is(input3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(conv2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(maxpool2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
         end select

      end select

   end subroutine backward_1d


   pure module subroutine backward_3d(self, previous, gradient)
      implicit none
      class(layer), intent(in out) :: self
      class(layer), intent(in) :: previous
      real, intent(in) :: gradient(:,:,:)

      ! Backward pass from a 3-d layer downstream currently implemented
      ! only for conv2d and reshape3d layers
      select type(this_layer => self % p)

       type is(conv2d_layer)

         ! Upstream layers permitted: conv2d, input3d, maxpool2d, reshape3d
         select type(prev_layer => previous % p)
          type is(maxpool2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(input3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(conv2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(reshape3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
         end select

       type is(maxpool2d_layer)

         ! Upstream layers permitted: conv2d, input3d, maxpool2d, reshape3d
         select type(prev_layer => previous % p)
          type is(conv2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(maxpool2d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(input3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(reshape3d_layer)
            call this_layer % backward(prev_layer % output, gradient)
         end select

       type is(reshape3d_layer)

         ! Upstream layers permitted: input1d, dense, flatten
         select type(prev_layer => previous % p)
          type is(input1d_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(dense_layer)
            call this_layer % backward(prev_layer % output, gradient)
          type is(flatten_layer)
            call this_layer % backward(prev_layer % output, gradient)
         end select

      end select

   end subroutine backward_3d


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

         ! Upstream layers permitted: input3d, conv2d, maxpool2d, reshape3d
         select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(reshape3d_layer)
            call this_layer % forward(prev_layer % output)
         end select

       type is(maxpool2d_layer)

         ! Upstream layers permitted: input3d, conv2d, maxpool2d, reshape3d
         select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(reshape3d_layer)
            call this_layer % forward(prev_layer % output)
         end select

       type is(flatten_layer)

         ! Upstream layers permitted: input3d, conv2d, maxpool2d, reshape3d
         select type(prev_layer => input % p)
          type is(input3d_layer)
            call this_layer % forward(prev_layer % output)
          type is(conv2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(maxpool2d_layer)
            call this_layer % forward(prev_layer % output)
          type is(reshape3d_layer)
            call this_layer % forward(prev_layer % output)
         end select

       type is(reshape3d_layer)

         ! Upstream layers permitted: input1d, dense, flatten
         select type(prev_layer => input % p)
          type is(input1d_layer)
            call this_layer % forward(prev_layer % output)
          type is(dense_layer)
            call this_layer % forward(prev_layer % output)
          type is(flatten_layer)
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
       type is(reshape3d_layer)
         allocate(output, source=this_layer % output)
       class default
         error stop '3-d output can only be read from a conv2d, input3d, maxpool2d, or reshape3d layer.'

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

   pure module function get_num_params(self) result(num_params)
      implicit none
      class(layer), intent(in) :: self
      integer :: num_params

      select type(this_layer => self % p)
       type is(input1d_layer)
         num_params = 0
       type is(input3d_layer)
         num_params = 0
       type is(dense_layer)
         num_params = this_layer % get_num_params()
       type is(conv2d_layer)
         num_params = this_layer % get_num_params()
       type is(maxpool2d_layer)
         num_params = 0
       type is(flatten_layer)
         num_params = 0
       type is(reshape3d_layer)
         num_params = 0
       class default
         error stop 'Unknown layer type.'
      end select

      impure module subroutine get_parameters(self, params)
         class(layer), intent(in) :: self
         real, allocatable, intent(inout) :: params(:)

         select type(this_layer => self % p)
          type is(input1d_layer)
            ! No parameters to get.
          type is(input3d_layer)
            ! No parameters to get.
          type is(dense_layer)
            call this_layer % get_parameters(params)
          type is(conv2d_layer)
            call this_layer % get_parameters(params)
          type is(maxpool2d_layer)
            ! No parameters to get.
          type is(flatten_layer)
            ! No parameters to get.
          type is(reshape3d_layer)
            ! No parameters to get.
          class default
            error stop 'Unknown layer type.'
         end select
      end subroutine get_parameters

   end function get_num_params

   impure elemental module subroutine update(self, learning_rate)
      implicit none
      class(layer), intent(in out) :: self
      real, intent(in) :: learning_rate

      select type(this_layer => self % p); type is(dense_layer)
         call this_layer % update(learning_rate)
      end select

   end subroutine update

end submodule nf_layer_submodule
