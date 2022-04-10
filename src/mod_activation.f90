module mod_activation

  !! A collection of activation functions and their derivatives.

  use mod_kinds, only: ik, rk

  implicit none

  private

  public :: activation_function
  public :: gaussian, gaussian_prime
  public :: relu, relu_prime
  public :: sigmoid, sigmoid_prime
  public :: step, step_prime
  public :: tanhf, tanh_prime

  interface

    pure function activation_function(x)
      import :: rk
      real(rk), intent(in) :: x(:)
      real(rk) :: activation_function(size(x))
    end function activation_function

    pure module function gaussian(x) result(res)
      !! Gaussian activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function gaussian

    pure module function gaussian_prime(x) result(res)
      !! First derivative of the Gaussian activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function gaussian_prime

    pure module function relu(x) result(res)
      !! REctified Linear Unit (RELU) activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function relu

    pure module function relu_prime(x) result(res)
      !! First derivative of the REctified Linear Unit (RELU) activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function relu_prime

    pure module function sigmoid(x) result(res)
      !! Sigmoid activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function sigmoid

    pure module function sigmoid_prime(x) result(res)
      !! First derivative of the sigmoid activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function sigmoid_prime

    pure module function step(x) result(res)
      !! Step activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function step

    pure module function step_prime(x) result(res)
      !! First derivative of the step activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function step_prime

    pure module function tanhf(x) result(res)
      !! Tangent hyperbolic activation function. 
      !! Same as the intrinsic tanh, but must be 
      !! defined here so that we can use procedure
      !! pointer with it.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function tanhf

    pure module function tanh_prime(x) result(res)
      !! First derivative of the tanh activation function.
      implicit none
      real(rk), intent(in) :: x(:)
      real(rk) :: res(size(x))
    end function tanh_prime

  end interface

end module mod_activation
