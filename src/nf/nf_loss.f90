module nf_loss

  !! This module will eventually provide a collection of loss functions and
  !! their derivatives. For the time being it provides only the quadratic
  !! function.

  implicit none

  private
  public :: quadratic, quadratic_derivative

  interface

    pure module function quadratic(true, predicted) result(res)
      !! Quadratic loss function:
      !!
      !!   L  = (predicted - true)**2 / 2
      !!
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res(size(true))
        !! Resulting loss values
    end function quadratic

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
