submodule(mod_activation) mod_activation_submodule

  !! A collection of activation functions and their derivatives.

  implicit none

contains

  pure module function gaussian(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = exp(-x**2)
  end function gaussian

  pure module function gaussian_prime(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = -2 * x * gaussian(x)
  end function gaussian_prime

  pure module function relu(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = max(0., x)
  end function relu

  pure module function relu_prime(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function relu_prime

  pure module function sigmoid(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = 1 / (1 + exp(-x))
  endfunction sigmoid

  pure module function sigmoid_prime(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = sigmoid(x) * (1 - sigmoid(x))
  end function sigmoid_prime

  pure module function step(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function step

  pure module function step_prime(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = 0
  end function step_prime

  pure module function tanhf(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = tanh(x)
  end function tanhf

  pure module function tanh_prime(x) result(res)
    real(rk), intent(in) :: x(:)
    real(rk) :: res(size(x))
    res = 1 - tanh(x)**2
  end function tanh_prime

end submodule mod_activation_submodule
