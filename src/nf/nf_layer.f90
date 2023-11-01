module nf_layer

  !! This module provides the `layer` type that is part of the public
  !! user-facing API.

  use nf_base_layer, only: base_layer
  use nf_optimizers, only: optimizer_base_type

  implicit none

  private
  public :: layer

  type :: layer

    !! Main layer type. Use custom constructor functions from
    !! nf_layer_constructors.f90 to create `layer` instances.

    class(base_layer), allocatable :: p
    character(:), allocatable :: name
    character(:), allocatable :: activation
    integer, allocatable :: layer_shape(:)
    integer, allocatable :: input_layer_shape(:)
    logical :: initialized = .false.

  contains

    procedure :: forward
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params
    procedure :: init
    procedure :: print_info
    procedure :: reset

    ! Specific subroutines for different array ranks
    procedure, private :: backward_1d
    procedure, private :: backward_3d
    procedure, private :: get_output_1d
    procedure, private :: get_output_3d

    generic :: backward => backward_1d, backward_3d
    generic :: get_output => get_output_1d, get_output_3d

  end type layer

  interface backward

    pure module subroutine backward_1d(self, previous, gradient)
      !! Apply a backward pass on the layer.
      !! This changes the internal state of the layer.
      !! This is normally called internally by the `network % backward`
      !! method.
      class(layer), intent(in out) :: self
        !! Layer instance
      class(layer), intent(in) :: previous
        !! Previous layer instance
      real, intent(in) :: gradient(:)
        !! Array of gradient values from the next layer
    end subroutine backward_1d

    pure module subroutine backward_3d(self, previous, gradient)
      !! Apply a backward pass on the layer.
      !! This changes the internal state of the layer.
      !! This is normally called internally by the `network % backward`
      !! method.
      class(layer), intent(in out) :: self
        !! Layer instance
      class(layer), intent(in) :: previous
        !! Previous layer instance
      real, intent(in) :: gradient(:,:,:)
        !! Array of gradient values from the next layer
    end subroutine backward_3d

  end interface backward

  interface

    pure module subroutine forward(self, input)
      !! Apply a forward pass on the layer.
      !! This changes the internal state of the layer.
      !! This is normally called internally by the `network % forward`
      !! method.
      class(layer), intent(in out) :: self
        !! Layer instance
      class(layer), intent(in) :: input
        !! Input layer instance
    end subroutine forward

    pure module subroutine get_output_1d(self, output)
      !! Returns the output values (activations) from this layer.
      class(layer), intent(in) :: self
        !! Layer instance
      real, allocatable, intent(out) :: output(:)
        !! Output values from this layer
    end subroutine get_output_1d

    pure module subroutine get_output_3d(self, output)
      !! Returns the output values (activations) from a layer with a 3-d output
      !! (e.g. input3d, conv2d)
      class(layer), intent(in) :: self
        !! Layer instance
      real, allocatable, intent(out) :: output(:,:,:)
        !! Output values from this layer
    end subroutine get_output_3d

    impure elemental module subroutine init(self, input)
      !! Initialize the layer, using information from the input layer,
      !! i.e. the layer that precedes this one.
      class(layer), intent(in out) :: self
        !! Layer instance
      class(layer), intent(in) :: input
        !! Input layer instance
    end subroutine init

    impure elemental module subroutine print_info(self)
      !! Prints a summary information about this layer to the screen.
      !! This method is called by `network % print_info` for all layers
      !! on that network.
      class(layer), intent(in) :: self
        !! Layer instance
    end subroutine print_info

    elemental module function get_num_params(self) result(num_params)
      !! Returns the number of parameters in this layer.
      class(layer), intent(in) :: self
        !! Layer instance
      integer :: num_params
        !! Number of parameters in this layer
    end function get_num_params

    module function get_params(self) result(params)
      !! Returns the parameters of this layer.
      class(layer), intent(in) :: self
        !! Layer instance
      real, allocatable :: params(:)
        !! Parameters of this layer
    end function get_params

    module function get_gradients(self) result(gradients)
      !! Returns the gradients of this layer.
      class(layer), intent(in) :: self
        !! Layer instance
      real, allocatable :: gradients(:)
        !! Gradients of this layer
    end function get_gradients

    module subroutine set_params(self, params)
      !! Returns the parameters of this layer.
      class(layer), intent(in out) :: self
        !! Layer instance
      real, intent(in) :: params(:)
        !! Parameters of this layer
    end subroutine set_params

    module subroutine reset(self)
      class(layer), intent(in out) :: self
    end subroutine reset

  end interface

end module nf_layer
