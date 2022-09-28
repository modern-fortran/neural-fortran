submodule(nf_loss) nf_loss_submodule

  implicit none

contains

  pure module function quadratic(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res(size(true))
    res = (predicted - true)**2 / 2
  end function quadratic

  pure module function quadratic_derivative(true, predicted) result(res)
    real, intent(in) :: true(:)
    real, intent(in) :: predicted(:)
    real :: res(size(true))
    res = predicted - true
  end function quadratic_derivative

end submodule nf_loss_submodule
