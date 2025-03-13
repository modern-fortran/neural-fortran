module nf_locally_connected1d_layer
    !! This modules provides a 1-d convolutional `locally_connected1d` type.
  
    use nf_activation, only: activation_function
    use nf_base_layer, only: base_layer
    implicit none
  
    private
    public :: locally_connected1d_layer
  
    type, extends(base_layer) :: locally_connected1d_layer
  
      integer :: width
      integer :: height
      integer :: channels
      integer :: kernel_size
      integer :: filters
  
      real, allocatable :: biases(:,:) ! size(filters)
      real, allocatable :: kernel(:,:,:,:) ! filters x channels x window x window
      real, allocatable :: output(:,:) ! filters x output_width * output_height
      real, allocatable :: z(:,:) ! kernel .dot. input + bias
  
      real, allocatable :: dw(:,:,:,:) ! weight (kernel) gradients
      real, allocatable :: db(:,:) ! bias gradients
      real, allocatable :: gradient(:,:)
  
      class(activation_function), allocatable :: activation
  
    contains
  
      procedure :: forward
      procedure :: backward
      procedure :: get_gradients
      procedure :: get_num_params
      procedure :: get_params
      procedure :: init
      procedure :: set_params
  
    end type locally_connected1d_layer
  
    interface locally_connected1d_layer
      module function locally_connected1d_layer_cons(filters, kernel_size, activation) &
        result(res)
        !! `locally_connected1d_layer` constructor function
        integer, intent(in) :: filters
        integer, intent(in) :: kernel_size
        class(activation_function), intent(in) :: activation
        type(locally_connected1d_layer) :: res
      end function locally_connected1d_layer_cons
    end interface locally_connected1d_layer
  
    interface
  
      module subroutine init(self, input_shape)
        !! Initialize the layer data structures.
        !!
        !! This is a deferred procedure from the `base_layer` abstract type.
        class(locally_connected1d_layer), intent(in out) :: self
          !! A `locally_connected1d_layer` instance
        integer, intent(in) :: input_shape(:)
          !! Input layer dimensions
      end subroutine init
  
      pure module subroutine forward(self, input)
        !! Apply a forward pass on the `locally_connected1d` layer.
        class(locally_connected1d_layer), intent(in out) :: self
          !! A `locally_connected1d_layer` instance
        real, intent(in) :: input(:,:)
          !! Input data
      end subroutine forward
  
      pure module subroutine backward(self, input, gradient)
        !! Apply a backward pass on the `locally_connected1d` layer.
        class(locally_connected1d_layer), intent(in out) :: self
          !! A `locally_connected1d_layer` instance
        real, intent(in) :: input(:,:)
          !! Input data (previous layer)
        real, intent(in) :: gradient(:,:)
          !! Gradient (next layer)
      end subroutine backward
  
      pure module function get_num_params(self) result(num_params)
        !! Get the number of parameters in the layer.
        class(locally_connected1d_layer), intent(in) :: self
          !! A `locally_connected1d_layer` instance
        integer :: num_params
          !! Number of parameters
      end function get_num_params
  
      module function get_params(self) result(params)
        !! Return the parameters (weights and biases) of this layer.
        !! The parameters are ordered as weights first, biases second.
        class(locally_connected1d_layer), intent(in), target :: self
          !! A `locally_connected1d_layer` instance
        real, allocatable :: params(:)
          !! Parameters to get
      end function get_params
  
      module function get_gradients(self) result(gradients)
        !! Return the gradients of this layer.
        !! The gradients are ordered as weights first, biases second.
        class(locally_connected1d_layer), intent(in), target :: self
          !! A `locally_connected1d_layer` instance
        real, allocatable :: gradients(:)
          !! Gradients to get
      end function get_gradients
  
      module subroutine set_params(self, params)
        !! Set the parameters of the layer.
        class(locally_connected1d_layer), intent(in out) :: self
          !! A `locally_connected1d_layer` instance
        real, intent(in) :: params(:)
          !! Parameters to set
      end subroutine set_params
  
    end interface

end module nf_locally_connected1d_layer
