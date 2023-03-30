module nf_activation_1d

  ! A collection of activation functions and their derivatives.

  implicit none

  private

  public :: activation_function
  public :: elu
  public :: exponential
  public :: gaussian
  public :: linear
  public :: relu
  public :: leaky_relu
  public :: sigmoid
  public :: softmax
  public :: softplus
  public :: step
  public :: tanhf

  type, abstract :: activation_function
  contains
    procedure(eval_i), deferred :: eval
    procedure(eval_i), deferred :: eval_prime
  end type activation_function

  abstract interface
    pure function eval_i(self, x) result(res)
      import :: activation_function
      class(activation_function), intent(in) :: self
      real, intent(in) :: x(:)
      real :: res(size(x))
    end function eval_i
  end interface

  type, extends(activation_function) :: elu
    real :: alpha
  contains
    procedure :: eval       => eval_elu
    procedure :: eval_prime => eval_elu_prime
  end type elu

  type, extends(activation_function) :: exponential
  contains
    procedure :: eval       => eval_exponential
    procedure :: eval_prime => eval_exponential
  end type exponential

  type, extends(activation_function) :: gaussian
  contains
    procedure :: eval       => eval_gaussian
    procedure :: eval_prime => eval_gaussian_prime
  end type gaussian

  type, extends(activation_function) :: linear
  contains
    procedure :: eval       => eval_linear
    procedure :: eval_prime => eval_linear_prime
  end type linear

  type, extends(activation_function) :: relu
  contains
    procedure :: eval       => eval_relu
    procedure :: eval_prime => eval_relu_prime
  end type relu

  type, extends(activation_function) :: leaky_relu
    real :: alpha
  contains
    procedure :: eval       => eval_leaky_relu
    procedure :: eval_prime => eval_leaky_relu_prime
  end type leaky_relu

  type, extends(activation_function) :: sigmoid
  contains
    procedure :: eval       => eval_sigmoid
    procedure :: eval_prime => eval_sigmoid_prime
  end type sigmoid

  type, extends(activation_function) :: softmax
  contains
    procedure :: eval       => eval_softmax
    procedure :: eval_prime => eval_softmax_prime
  end type softmax

  type, extends(activation_function) :: softplus
  contains
    procedure :: eval       => eval_softplus
    procedure :: eval_prime => eval_softplus_prime
  end type softplus

  type, extends(activation_function) :: step
  contains
    procedure :: eval       => eval_step
    procedure :: eval_prime => eval_step_prime
  end type step

  type, extends(activation_function) :: tanhf
  contains
    procedure :: eval       => eval_tanh
    procedure :: eval_prime => eval_tanh_prime
  end type tanhf

contains

  pure function eval_elu(self, x) result(res)
    ! Exponential Linear Unit (ELU) activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x >= 0)
      res = x
    elsewhere
      res = self%alpha * (exp(x) - 1)
    end where
  end function eval_elu

  pure function eval_elu_prime(self, x) result(res)
    ! First derivative of the Exponential Linear Unit (ELU)
    ! activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x >= 0)
      res = 1
    elsewhere
      res = self%alpha * exp(x)
    end where
  end function eval_elu_prime

  pure function eval_exponential(self, x) result(res)
    ! Exponential activation function.
    class(exponential), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x)
  end function eval_exponential

  pure function eval_gaussian(self, x) result(res)
    ! Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(-x**2)
  end function eval_gaussian

  pure function eval_gaussian_prime(self, x) result(res)
    ! First derivative of the Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = -2 * x * self % eval(x)
  end function eval_gaussian_prime

  pure function eval_linear(self, x) result(res)
    ! Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = x
  end function eval_linear

  pure function eval_linear_prime(self, x) result(res)
    ! First derivative of the Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1
  end function eval_linear_prime

  pure function eval_relu(self, x) result(res)
    !! Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = max(0., x)
  end function eval_relu

  pure function eval_relu_prime(self, x) result(res)
    ! First derivative of the Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_relu_prime

  pure function eval_leaky_relu(self, x) result(res)
    !! Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = max(self%alpha*x, x)
  end function eval_leaky_relu

  pure function eval_leaky_relu_prime(self, x) result(res)
    ! First derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = self%alpha
    end where
  end function eval_leaky_relu_prime

  pure function eval_sigmoid(self, x) result(res)
    ! Sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1 / (1 + exp(-x))
  endfunction eval_sigmoid

  pure function eval_sigmoid_prime(self, x) result(res)
    ! First derivative of the sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = self % eval(x) * (1 - self % eval(x))
  end function eval_sigmoid_prime

  pure function eval_softmax(self, x) result(res)
    !! Softmax activation function
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x - maxval(x))
    res = res / sum(res)
  end function eval_softmax

  pure function eval_softmax_prime(self, x) result(res)
    !! Derivative of the softmax activation function.
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = self%eval(x) * (1 - self%eval(x))
  end function eval_softmax_prime

  pure function eval_softplus(self, x) result(res)
    ! Softplus activation function.
    class(softplus), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = log(exp(x) + 1)
  end function eval_softplus

  pure function eval_softplus_prime(self, x) result(res)
      class(softplus), intent(in) :: self
    ! First derivative of the softplus activation function.
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x) / (exp(x) + 1)
  end function eval_softplus_prime

  pure function eval_step(self, x) result(res)
    ! Step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_step

  pure function eval_step_prime(self, x) result(res)
    ! First derivative of the step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 0
  end function eval_step_prime

  pure function eval_tanh(self, x) result(res)
    ! Tangent hyperbolic activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = tanh(x)
  end function eval_tanh

  pure function eval_tanh_prime(self, x) result(res)
    ! First derivative of the tanh activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1 - tanh(x)**2
  end function eval_tanh_prime

end module nf_activation_1d
