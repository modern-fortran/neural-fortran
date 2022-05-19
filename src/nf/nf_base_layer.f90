module nf_base_layer

  !! This module provides the abstract base layer type, to be extended by
  !! specific concrete types.

  use nf_activation, only: activation_function

  implicit none

  private
  public :: base_layer

  type, abstract :: base_layer

    !! This type is the base for creating concrete layer instances.
    !! Extend this type when creating other concrete layer types.

    character(:), allocatable :: activation_name
    procedure(activation_function), pointer, nopass :: &
      activation => null()
    procedure(activation_function), pointer, nopass :: &
      activation_prime => null()

  contains

    procedure(init_interface), deferred :: init
    procedure :: set_activation

  end type base_layer

  abstract interface
    subroutine init_interface(self, input_shape)
      !! Initialize the internal layer data structures.
      import :: base_layer
      class(base_layer), intent(in out) :: self
        !! Layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer, i.e. the layer that precedes
        !! this layer
    end subroutine init_interface
  end interface

  interface
    elemental module subroutine set_activation(self, activation)
      !! Set the activation functions.
      class(base_layer), intent(in out) :: self
        !! Layer instance
      character(*), intent(in) :: activation
        !! String with the activation function name
    end subroutine set_activation
  end interface

end module nf_base_layer
