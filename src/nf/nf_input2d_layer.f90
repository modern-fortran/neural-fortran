module nf_input2d_layer

  !! This module provides the `input2d_layer` type.

  use nf_base_layer, only: base_layer
  implicit none

  private
  public :: input2d_layer

  type, extends(base_layer) :: input2d_layer
    real, allocatable :: output(:,:)
  contains
    procedure :: init
    procedure :: set
  end type input2d_layer

  interface input2d_layer
    pure module function input2d_layer_cons(output_shape) result(res)
      !! Create a new instance of the 2-d input layer.
      !! Only used internally by the `layer % init` method.
      integer, intent(in) :: output_shape(2)
        !! Shape of the input layer
      type(input2d_layer) :: res
        !! 2-d input layer instance
    end function input2d_layer_cons
  end interface input2d_layer

  interface

    module subroutine init(self, input_shape)
      !! Only here to satisfy the language rules
      !! about deferred methods of abstract types.
      !! This method does nothing for this type and should not be called.
      class(input2d_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module subroutine set(self, values)
      class(input2d_layer), intent(in out) :: self
        !! Layer instance
      real, intent(in) :: values(:,:)
        !! Values to set
    end subroutine set

  end interface

end module nf_input2d_layer 