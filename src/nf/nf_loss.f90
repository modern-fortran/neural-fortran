module nf_loss

  !! This module will eventually provide a collection of loss functions and
  !! their derivatives. For the time being it provides only the quadratic
  !! function.

  implicit none

  private
  public :: loss_derivative_interface
  public :: mse, mse_derivative
  public :: quadratic, quadratic_derivative

  interface

    pure function loss_derivative_interface(true, predicted) result(res)
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res(size(true))
        !! Resulting loss values
    end function loss_derivative_interface

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

    pure module function mse(true, predicted) result(res)
      !! Mean square error loss function:
      !!
      !!   L  = (predicted - true)**2 / n
      !!
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res(size(true))
        !! Resulting loss values
    end function mse

    pure module function mse_derivative(true, predicted) result(res)
      !! First derivative of the quadratic loss function:
      !!
      !!   L' =  2 * (predicted - true) / n
      !!
      real, intent(in) :: true(:)
        !! True values, i.e. labels from training datasets
      real, intent(in) :: predicted(:)
        !! Values predicted by the network
      real :: res(size(true))
        !! Resulting loss values
    end function mse_derivative

  end interface

end module nf_loss
