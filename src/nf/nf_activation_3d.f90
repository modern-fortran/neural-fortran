module nf_activation_3d

  ! A collection of activation functions and their derivatives.

  implicit none

  private

  public :: activation_function
  public :: elu, elu_prime
  public :: exponential
  public :: gaussian, gaussian_prime
  public :: linear, linear_prime
  public :: relu, relu_prime
  public :: sigmoid, sigmoid_prime
  public :: softmax, softmax_prime
  public :: softplus, softplus_prime
  public :: step, step_prime
  public :: tanhf, tanh_prime

  interface
    pure function activation_function(x) result(res)
      real, intent(in) :: x(:,:,:)
      real :: res(size(x,1),size(x,2),size(x,3))
    end function activation_function
  end interface

contains

  pure function elu(x, alpha) result(res)
    ! Exponential Linear Unit (ELU) activation function.
    real, intent(in) :: x(:,:,:)
    real, intent(in) :: alpha
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x >= 0)
      res = x
    elsewhere
      res = alpha * (exp(x) - 1)
    end where
  end function elu

  pure function elu_prime(x, alpha) result(res)
    ! First derivative of the Exponential Linear Unit (ELU)
    ! activation function.
    real, intent(in) :: x(:,:,:)
    real, intent(in) :: alpha
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x >= 0)
      res = 1
    elsewhere
      res = alpha * exp(x)
    end where
  end function elu_prime

  pure function exponential(x) result(res)
    ! Exponential activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x)
  end function exponential

  pure function gaussian(x) result(res)
    ! Gaussian activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(-x**2)
  end function gaussian

  pure function gaussian_prime(x) result(res)
    ! First derivative of the Gaussian activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = -2 * x * gaussian(x)
  end function gaussian_prime

  pure function linear(x) result(res)
    ! Linear activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = x
  end function linear

  pure function linear_prime(x) result(res)
    ! First derivative of the linear activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1
  end function linear_prime

  pure function relu(x) result(res)
    !! Rectified Linear Unit (ReLU) activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = max(0., x)
  end function relu

  pure function relu_prime(x) result(res)
    ! First derivative of the Rectified Linear Unit (ReLU) activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function relu_prime

  pure function sigmoid(x) result(res)
    ! Sigmoid activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1 / (1 + exp(-x))
  endfunction sigmoid

  pure function sigmoid_prime(x) result(res)
    ! First derivative of the sigmoid activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = sigmoid(x) * (1 - sigmoid(x))
  end function sigmoid_prime

  pure function softmax(x) result(res)
    !! Softmax activation function
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x - maxval(x))
    res = res / sum(res)
  end function softmax

  pure function softmax_prime(x) result(res)
    !! Derivative of the softmax activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = softmax(x) * (1 - softmax(x))
  end function softmax_prime

  pure function softplus(x) result(res)
    ! Softplus activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = log(exp(x) + 1)
  end function softplus

  pure function softplus_prime(x) result(res)
    ! First derivative of the softplus activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x) / (exp(x) + 1)
  end function softplus_prime

  pure function step(x) result(res)
    ! Step activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function step

  pure function step_prime(x) result(res)
    ! First derivative of the step activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 0
  end function step_prime

  pure function tanhf(x) result(res)
    ! Tangent hyperbolic activation function. 
    ! Same as the intrinsic tanh, but must be 
    ! defined here so that we can use procedure
    ! pointer with it.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = tanh(x)
  end function tanhf

  pure function tanh_prime(x) result(res)
    ! First derivative of the tanh activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1 - tanh(x)**2
  end function tanh_prime

end module nf_activation_3d
