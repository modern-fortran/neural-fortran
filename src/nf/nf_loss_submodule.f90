submodule(nf_loss) nf_loss_submodule

  implicit none

contains

  pure module function quadratic_eval(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res
    res = sum((predicted - true)**2) / 2
  end function quadratic_eval

  pure module function quadratic_derivative(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res(size(true))
    res = predicted - true
  end function quadratic_derivative

  pure module function mse_eval(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res
    res = sum((predicted - true)**2) / size(true)
  end function mse_eval

  pure module function mse_derivative(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res(size(true))
    res = 2 * (predicted - true) / size(true)
  end function mse_derivative

end submodule nf_loss_submodule
