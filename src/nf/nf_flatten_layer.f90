module nf_flatten_layer

  !! This module provides the concrete flatten layer type.
  !! It is used internally by the layer type.
  !! It is not intended to be used directly by the user.

  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: flatten_layer

  type, extends(base_layer) :: flatten_layer

    !! Concrete implementation of a flatten (3-d to 1-d) layer.

    integer, allocatable :: input_shape(:)
    integer :: output_size

    real, allocatable :: gradient_2d(:,:)
    real, allocatable :: gradient_3d(:,:,:)
    real, allocatable :: output(:)

  contains

    procedure :: backward_2d
    procedure :: backward_3d
    generic :: backward => backward_2d, backward_3d

    procedure :: forward_2d
    procedure :: forward_3d
    generic :: forward => forward_2d, forward_3d

    procedure :: init

  end type flatten_layer

  interface flatten_layer
    elemental module function flatten_layer_cons() result(res)
      !! This function returns the `flatten_layer` instance.
      type(flatten_layer) :: res
        !! `flatten_layer` instance
    end function flatten_layer_cons
  end interface flatten_layer

  interface

    pure module subroutine backward_2d(self, input, gradient)
      !! Apply the backward pass to the flatten layer for 2D input.
      !! This is a reshape operation from 1-d gradient to 2-d input.
      class(flatten_layer), intent(in out) :: self
        !! Flatten layer instance
      real, intent(in) :: input(:,:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward_2d

    pure module subroutine backward_3d(self, input, gradient)
      !! Apply the backward pass to the flatten layer for 3D input.
      !! This is a reshape operation from 1-d gradient to 3-d input.
      class(flatten_layer), intent(in out) :: self
        !! Flatten layer instance
      real, intent(in) :: input(:,:,:)
        !! Input from the previous layer
      real, intent(in) :: gradient(:)
        !! Gradient from the next layer
    end subroutine backward_3d

    pure module subroutine forward_2d(self, input)
      !! Propagate forward the layer for 2D input.
      !! Calling this subroutine updates the values of a few data components
      !! of `flatten_layer` that are needed for the backward pass.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:,:)
        !! Input from the previous layer
    end subroutine forward_2d

    pure module subroutine forward_3d(self, input)
      !! Propagate forward the layer for 3D input.
      !! Calling this subroutine updates the values of a few data components
      !! of `flatten_layer` that are needed for the backward pass.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      real, intent(in) :: input(:,:,:)
        !! Input from the previous layer
    end subroutine forward_3d

    module subroutine init(self, input_shape)
      !! Initialize the layer data structures.
      !!
      !! This is a deferred procedure from the `base_layer` abstract type.
      class(flatten_layer), intent(in out) :: self
        !! Dense layer instance
      integer, intent(in) :: input_shape(:)
        !! Shape of the input layer
    end subroutine init

  end interface

end module nf_flatten_layer
