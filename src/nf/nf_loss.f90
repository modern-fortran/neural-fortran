module nf_loss

  !! This module will eventually provide a collection of loss functions and
  !! their derivatives. For the time being it provides only the quadratic
  !! function.

  implicit none

  private
  public :: loss_type
  public :: quadratic

  type, abstract :: loss_type
  contains
    procedure(loss_interface), nopass, deferred :: eval
    procedure(loss_derivative_interface), nopass, deferred :: derivative
  end type loss_type

  abstract interface
    pure function loss_interface(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res
    end function loss_interface
    pure function loss_derivative_interface(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res(size(true))
    end function loss_derivative_interface
  end interface

  type, extends(loss_type) :: quadratic
  contains
    procedure, nopass :: eval => quadratic_eval
    procedure, nopass :: derivative => quadratic_derivative
  end type quadratic

  interface

    pure module function quadratic_eval(true, predicted) result(res)
      !! Quadratic loss function:
      !!
      !!   L  = (predicted - true)**2 / 2
      !!
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res
        !! Resulting loss values
    end function quadratic_eval

    pure module function quadratic_derivative(true, predicted) result(res)
      !! First derivative of the quadratic loss function:
      !!
      !!   L' =  predicted - true
      !!
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res(size(true))
        !! Resulting loss values
    end function quadratic_derivative

  end interface

end module nf_loss
