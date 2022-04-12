module mod_layer

  !! Defines the layer type and its methods.

  use mod_activation
  use mod_kinds, only: ik, rk

  implicit none

  private
  public :: array1d, array2d, db_init, db_co_sum, dw_init, dw_co_sum, layer_type

  type :: layer_type
    real(rk), allocatable :: a(:) !! activations
    real(rk), allocatable :: b(:) !! biases
    real(rk), allocatable :: w(:,:) !! weights
    real(rk), allocatable :: z(:) !! arg. to activation function
    procedure(activation_function), pointer, nopass :: activation => null()
    procedure(activation_function), pointer, nopass :: activation_prime => null()
    character(len=:), allocatable :: activation_str !! activation character string
  contains
    procedure, public, pass(self) :: set_activation
  end type layer_type

  type :: array1d
    real(rk), allocatable :: array(:)
  end type array1d

  type :: array2d
    real(rk), allocatable :: array(:,:)
  end type array2d

  interface layer_type
    type(layer_type) module function constructor(this_size, next_size) result(layer)
      !! Layer class constructor. this_size is the number of neurons in the layer.
      !! next_size is the number of neurons in the next layer, used to allocate
      !! the weights.
      implicit none
      integer(ik), intent(in) :: this_size, next_size
    end function constructor
  end interface layer_type

  interface array1d
    pure type(array1d) module function array1d_constructor(length) result(a)
      !! Overloads the default type constructor.
      implicit none
      integer(ik), intent(in) :: length
    end function array1d_constructor  
  end interface array1d

  interface array2d  
    pure type(array2d) module function array2d_constructor(dims) result(a)
      !! Overloads the default type constructor.
      integer(ik), intent(in) :: dims(2)
    end function array2d_constructor
  end interface array2d
  
  interface

    pure module subroutine db_init(db, dims)
      !! Initialises biases structure.
      implicit none
      type(array1d), allocatable, intent(in out) :: db(:)
      integer(ik), intent(in) :: dims(:)
    end subroutine db_init  

    pure module subroutine dw_init(dw, dims)
      !! Initialises weights structure.
      implicit none
      type(array2d), allocatable, intent(in out) :: dw(:)
      integer(ik), intent(in) :: dims(:)
    end subroutine dw_init
    
    module subroutine db_co_sum(db)
      !! Performs a collective sum of bias tendencies.
      implicit none
      type(array1d), allocatable, intent(in out) :: db(:)
    end subroutine db_co_sum
    
    module subroutine dw_co_sum(dw)
      !! Performs a collective sum of weights tendencies.
      implicit none
      type(array2d), allocatable, intent(in out) :: dw(:)
    end subroutine dw_co_sum

    pure elemental module subroutine set_activation(self, activation)
      !! Sets the activation function. Input string must match one of
      !! provided activation functions, otherwise it defaults to sigmoid.
      !! If activation not present, defaults to sigmoid.
      implicit none
      class(layer_type), intent(in out) :: self
      character(len=*), intent(in) :: activation
    end subroutine set_activation
  
  end interface

end module mod_layer
