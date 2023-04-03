module nf_activation

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
    procedure(eval_1d_i), deferred :: eval_1d
    procedure(eval_1d_i), deferred :: eval_1d_prime
    procedure(eval_3d_i), deferred :: eval_3d
    procedure(eval_3d_i), deferred :: eval_3d_prime
    procedure :: get_name

    generic :: eval => eval_1d, eval_3d
    generic :: eval_prime => eval_1d_prime, eval_3d_prime

  end type activation_function

  abstract interface

    pure function eval_1d_i(self, x) result(res)
      import :: activation_function
      class(activation_function), intent(in) :: self
      real, intent(in) :: x(:)
      real :: res(size(x))
    end function eval_1d_i

    pure function eval_3d_i(self, x) result(res)
      import :: activation_function
      class(activation_function), intent(in) :: self
      real, intent(in) :: x(:,:,:)
      real :: res(size(x,1),size(x,2),size(x,3))
    end function eval_3d_i

  end interface

  type, extends(activation_function) :: elu
    real :: alpha = 1 ! Keras & PyTorch default
  contains
    procedure :: eval_1d       => eval_1d_elu
    procedure :: eval_1d_prime => eval_1d_elu_prime
    procedure :: eval_3d       => eval_3d_elu
    procedure :: eval_3d_prime => eval_3d_elu_prime
  end type elu

  type, extends(activation_function) :: exponential
  contains
    procedure :: eval_1d       => eval_1d_exponential
    procedure :: eval_1d_prime => eval_1d_exponential
    procedure :: eval_3d       => eval_3d_exponential
    procedure :: eval_3d_prime => eval_3d_exponential
  end type exponential

  type, extends(activation_function) :: gaussian
  contains
    procedure :: eval_1d       => eval_1d_gaussian
    procedure :: eval_1d_prime => eval_1d_gaussian_prime
    procedure :: eval_3d       => eval_3d_gaussian
    procedure :: eval_3d_prime => eval_3d_gaussian_prime
  end type gaussian

  type, extends(activation_function) :: linear
  contains
    procedure :: eval_1d       => eval_1d_linear
    procedure :: eval_1d_prime => eval_1d_linear_prime
    procedure :: eval_3d       => eval_3d_linear
    procedure :: eval_3d_prime => eval_3d_linear_prime
  end type linear

  type, extends(activation_function) :: relu
  contains
    procedure :: eval_1d       => eval_1d_relu
    procedure :: eval_1d_prime => eval_1d_relu_prime
    procedure :: eval_3d       => eval_3d_relu
    procedure :: eval_3d_prime => eval_3d_relu_prime
  end type relu

  type, extends(activation_function) :: leaky_relu
    real :: alpha = 0.3 ! Keras default (PyTorch default is 0.01)
  contains
    procedure :: eval_1d       => eval_1d_leaky_relu
    procedure :: eval_1d_prime => eval_1d_leaky_relu_prime
    procedure :: eval_3d       => eval_3d_leaky_relu
    procedure :: eval_3d_prime => eval_3d_leaky_relu_prime
  end type leaky_relu

  type, extends(activation_function) :: sigmoid
  contains
    procedure :: eval_1d       => eval_1d_sigmoid
    procedure :: eval_1d_prime => eval_1d_sigmoid_prime
    procedure :: eval_3d       => eval_3d_sigmoid
    procedure :: eval_3d_prime => eval_3d_sigmoid_prime
  end type sigmoid

  type, extends(activation_function) :: softmax
  contains
    procedure :: eval_1d       => eval_1d_softmax
    procedure :: eval_1d_prime => eval_1d_softmax_prime
    procedure :: eval_3d       => eval_3d_softmax
    procedure :: eval_3d_prime => eval_3d_softmax_prime
  end type softmax

  type, extends(activation_function) :: softplus
  contains
    procedure :: eval_1d       => eval_1d_softplus
    procedure :: eval_1d_prime => eval_1d_softplus_prime
    procedure :: eval_3d       => eval_3d_softplus
    procedure :: eval_3d_prime => eval_3d_softplus_prime
  end type softplus

  type, extends(activation_function) :: step
  contains
    procedure :: eval_1d       => eval_1d_step
    procedure :: eval_1d_prime => eval_1d_step_prime
    procedure :: eval_3d       => eval_3d_step
    procedure :: eval_3d_prime => eval_3d_step_prime
  end type step

  type, extends(activation_function) :: tanhf
  contains
    procedure :: eval_1d       => eval_1d_tanh
    procedure :: eval_1d_prime => eval_1d_tanh_prime
    procedure :: eval_3d       => eval_3d_tanh
    procedure :: eval_3d_prime => eval_3d_tanh_prime
  end type tanhf

contains

  pure function eval_1d_elu(self, x) result(res)
    ! Exponential Linear Unit (ELU) activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x >= 0)
      res = x
    elsewhere
      res = self % alpha * (exp(x) - 1)
    end where
  end function eval_1d_elu

  pure function eval_1d_elu_prime(self, x) result(res)
    ! First derivative of the Exponential Linear Unit (ELU)
    ! activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x >= 0)
      res = 1
    elsewhere
      res = self % alpha * exp(x)
    end where
  end function eval_1d_elu_prime

  pure function eval_3d_elu(self, x) result(res)
    ! Exponential Linear Unit (ELU) activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x >= 0)
      res = x
    elsewhere
      res = self % alpha * (exp(x) - 1)
    end where
  end function eval_3d_elu

  pure function eval_3d_elu_prime(self, x) result(res)
    ! First derivative of the Exponential Linear Unit (ELU)
    ! activation function.
    class(elu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x >= 0)
      res = 1
    elsewhere
      res = self % alpha * exp(x)
    end where
  end function eval_3d_elu_prime

  pure function eval_1d_exponential(self, x) result(res)
    ! Exponential activation function.
    class(exponential), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x)
  end function eval_1d_exponential

  pure function eval_3d_exponential(self, x) result(res)
    ! Exponential activation function.
    class(exponential), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x)
  end function eval_3d_exponential

  pure function eval_1d_gaussian(self, x) result(res)
    ! Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(-x**2)
  end function eval_1d_gaussian

  pure function eval_1d_gaussian_prime(self, x) result(res)
    ! First derivative of the Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = -2 * x * self % eval_1d(x)
  end function eval_1d_gaussian_prime

  pure function eval_3d_gaussian(self, x) result(res)
    ! Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(-x**2)
  end function eval_3d_gaussian

  pure function eval_3d_gaussian_prime(self, x) result(res)
    ! First derivative of the Gaussian activation function.
    class(gaussian), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = -2 * x * self % eval_3d(x)
  end function eval_3d_gaussian_prime

  pure function eval_1d_linear(self, x) result(res)
    ! Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = x
  end function eval_1d_linear

  pure function eval_1d_linear_prime(self, x) result(res)
    ! First derivative of the Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1
  end function eval_1d_linear_prime

  pure function eval_3d_linear(self, x) result(res)
    ! Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = x
  end function eval_3d_linear

  pure function eval_3d_linear_prime(self, x) result(res)
    ! First derivative of the Linear activation function.
    class(linear), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1
  end function eval_3d_linear_prime

  pure function eval_1d_relu(self, x) result(res)
    !! Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = max(0., x)
  end function eval_1d_relu

  pure function eval_1d_relu_prime(self, x) result(res)
    ! First derivative of the Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_1d_relu_prime

  pure function eval_3d_relu(self, x) result(res)
    !! Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = max(0., x)
  end function eval_3d_relu

  pure function eval_3d_relu_prime(self, x) result(res)
    ! First derivative of the Rectified Linear Unit (ReLU) activation function.
    class(relu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_3d_relu_prime

  pure function eval_1d_leaky_relu(self, x) result(res)
    !! Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = max(self % alpha * x, x)
  end function eval_1d_leaky_relu

  pure function eval_1d_leaky_relu_prime(self, x) result(res)
    ! First derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = self % alpha
    end where
  end function eval_1d_leaky_relu_prime

  pure function eval_3d_leaky_relu(self, x) result(res)
    !! Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = max(self % alpha * x, x)
  end function eval_3d_leaky_relu

  pure function eval_3d_leaky_relu_prime(self, x) result(res)
    ! First derivative of the Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    class(leaky_relu), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x > 0)
      res = 1
    elsewhere
      res = self % alpha
    end where
  end function eval_3d_leaky_relu_prime

  pure function eval_1d_sigmoid(self, x) result(res)
    ! Sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1 / (1 + exp(-x))
  endfunction eval_1d_sigmoid

  pure function eval_1d_sigmoid_prime(self, x) result(res)
    ! First derivative of the sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = self % eval_1d(x) * (1 - self % eval_1d(x))
  end function eval_1d_sigmoid_prime

  pure function eval_3d_sigmoid(self, x) result(res)
    ! Sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1 / (1 + exp(-x))
  endfunction eval_3d_sigmoid

  pure function eval_3d_sigmoid_prime(self, x) result(res)
    ! First derivative of the sigmoid activation function.
    class(sigmoid), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = self % eval_3d(x) * (1 - self % eval_3d(x))
  end function eval_3d_sigmoid_prime

  pure function eval_1d_softmax(self, x) result(res)
    !! Softmax activation function
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x - maxval(x))
    res = res / sum(res)
  end function eval_1d_softmax

  pure function eval_1d_softmax_prime(self, x) result(res)
    !! Derivative of the softmax activation function.
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = self%eval_1d(x) * (1 - self%eval_1d(x))
  end function eval_1d_softmax_prime

  pure function eval_3d_softmax(self, x) result(res)
    !! Softmax activation function
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x - maxval(x))
    res = res / sum(res)
  end function eval_3d_softmax

  pure function eval_3d_softmax_prime(self, x) result(res)
    !! Derivative of the softmax activation function.
    class(softmax), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = self % eval_3d(x) * (1 - self % eval_3d(x))
  end function eval_3d_softmax_prime

  pure function eval_1d_softplus(self, x) result(res)
    ! Softplus activation function.
    class(softplus), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = log(exp(x) + 1)
  end function eval_1d_softplus

  pure function eval_1d_softplus_prime(self, x) result(res)
      class(softplus), intent(in) :: self
    ! First derivative of the softplus activation function.
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = exp(x) / (exp(x) + 1)
  end function eval_1d_softplus_prime

  pure function eval_3d_softplus(self, x) result(res)
    ! Softplus activation function.
    class(softplus), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = log(exp(x) + 1)
  end function eval_3d_softplus

  pure function eval_3d_softplus_prime(self, x) result(res)
      class(softplus), intent(in) :: self
    ! First derivative of the softplus activation function.
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = exp(x) / (exp(x) + 1)
  end function eval_3d_softplus_prime

  pure function eval_1d_step(self, x) result(res)
    ! Step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_1d_step

  pure function eval_1d_step_prime(self, x) result(res)
    ! First derivative of the step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 0
  end function eval_1d_step_prime

  pure function eval_3d_step(self, x) result(res)
    ! Step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    where (x > 0)
      res = 1
    elsewhere
      res = 0
    end where
  end function eval_3d_step

  pure function eval_3d_step_prime(self, x) result(res)
    ! First derivative of the step activation function.
    class(step), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 0
  end function eval_3d_step_prime

  pure function eval_1d_tanh(self, x) result(res)
    ! Tangent hyperbolic activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = tanh(x)
  end function eval_1d_tanh

  pure function eval_1d_tanh_prime(self, x) result(res)
    ! First derivative of the tanh activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:)
    real :: res(size(x))
    res = 1 - tanh(x)**2
  end function eval_1d_tanh_prime

  pure function eval_3d_tanh(self, x) result(res)
    ! Tangent hyperbolic activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = tanh(x)
  end function eval_3d_tanh

  pure function eval_3d_tanh_prime(self, x) result(res)
    ! First derivative of the tanh activation function.
    class(tanhf), intent(in) :: self
    real, intent(in) :: x(:,:,:)
    real :: res(size(x,1),size(x,2),size(x,3))
    res = 1 - tanh(x)**2
  end function eval_3d_tanh_prime

  pure function get_name(self) result(name)
    !! Return the name of the activation function.
    !!
    !! Normally we would place this in the definition of each type, however
    !! accessing the name variable directly from the type would require type
    !! guards just like we have here. This at least keeps all the type guards
    !! in one place.
    class(activation_function), intent(in) :: self
      !! The activation function instance.
    character(:), allocatable :: name
      !! The name of the activation function.
    select type (self)
    class is (elu)
      name = 'elu'
    class is (exponential)
      name = 'exponential'
    class is (gaussian)
      name = 'gaussian'
    class is (linear)
      name = 'linear'
    class is (relu)
      name = 'relu'
    class is (leaky_relu)
      name = 'leaky_relu'
    class is (sigmoid)
      name = 'sigmoid'
    class is (softmax)
      name = 'softmax'
    class is (softplus)
      name = 'softplus'
    class is (step)
      name = 'step'
    class is (tanhf)
      name = 'tanh'
    class default
      error stop 'Unknown activation function type.'
    end select
  end function get_name

end module nf_activation
