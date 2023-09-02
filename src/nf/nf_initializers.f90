module nf_initializers
  implicit none

  private
  public :: initializer_type, glorot, he

  type, abstract :: initializer_type
  contains
    procedure(init), deferred :: init
  end type initializer_type

  abstract interface
    subroutine init(self, x)
      import :: initializer_type
      class(initializer_type), intent(in) :: self
      real, intent(inout) :: x(:)
    end subroutine init
  end interface

  type, extends(initializer_type) :: glorot
  contains
    procedure :: init => init_glorot
  end type glorot

  type, extends(initializer_type) :: he
  contains
    procedure :: init => init_he
  end type he

contains

  subroutine init_glorot(self, x)
    class(glorot), intent(in) :: self
    real, intent(inout) :: x(:)
    error stop 'Not implemented'
  end subroutine init_glorot

  subroutine init_he(self, x)
    class(he), intent(in) :: self
    real, intent(inout) :: x(:)
    error stop 'Not implemented'
  end subroutine init_he

end module nf_initializers