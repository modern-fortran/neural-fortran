module nf_layer

  !! This module provides the `layer` type that is part of the public
  !! user-facing API.

  use nf_base_layer, only: base_layer

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

    procedure :: backward
    procedure :: forward
    procedure :: init
    procedure :: print_info
    procedure :: update

    ! Specific output subroutines for different array ranks,
    ! available via generic `get_output`.
    procedure, private :: get_output_1d
    procedure, private :: get_output_3d

    generic :: get_output => get_output_1d, get_output_3d

  end type layer

  interface

    pure module subroutine backward(self, previous, gradient)
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
    end subroutine backward

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

    impure elemental module subroutine update(self, learning_rate)
      !! Update the weights and biases on the layer using the stored
      !! gradients (from backward passes), and flush those same stored
      !! gradients to zero.
      !! This changes the state of the layer.
      !! Typically used only internally from the `network % update` method.
      class(layer), intent(in out) :: self
        !! Layer instance
      real, intent(in) :: learning_rate
        !! Learning rate to use; must be > 0.
    end subroutine update

  end interface

end module nf_layer
