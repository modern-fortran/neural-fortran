module nf_optimizers

  !! This module provides optimizer types to pass to the network train or update
  !! methods. The implementation is based on an abstract optimizer base type
  !! which has a required minimize method. The minimize method is an elemental
  !! subroutine to allow operating in-place on arrays of network parameters
  !! (weights/kernels and biases) of arbitrary ranks. An implementation of a new
  !! optimizer thus requires writing a concrete optimizer type that extends the
  !! abstract optimizer base type, and that implements a concrete minimize
  !! method that accepts a scalar or array of network parameters to update and
  !! the corresponding loss gradients.

  implicit none

  private
  public :: optimizer_base_type, sgd, rmsprop, adam

  type, abstract :: optimizer_base_type
    real :: learning_rate = 0.01
  contains
    procedure(init), deferred :: init
    procedure(minimize), deferred :: minimize
  end type optimizer_base_type

  abstract interface

    impure elemental subroutine init(self, num_params)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(inout) :: self
      integer, intent(in) :: num_params
    end subroutine init

    pure subroutine minimize(self, param, gradient)
      import :: optimizer_base_type
      class(optimizer_base_type), intent(inout) :: self
      real, intent(inout) :: param(:)
      real, intent(in) :: gradient(:)
    end subroutine minimize

  end interface

  type, extends(optimizer_base_type) :: sgd
    !! Stochastic Gradient Descent optimizer
    real :: momentum = 0
    logical :: nesterov = .false.
    real, allocatable, private :: velocity(:)
  contains
    procedure :: init => init_sgd
    procedure :: minimize => minimize_sgd
  end type sgd

  type, extends(optimizer_base_type) :: rmsprop
    !! RMSProp optimizer by Hinton et al. (2012)
    !!
    !! Hinton, G., Srivastava, N. and Swersky, K., 2012. Neural networks for
    !! machine learning lecture 6a overview of mini-batch gradient descent.
    !! Cited on 2023-07-19, 14(8), p.2. Available at:
    !! http://www.cs.toronto.edu/~hinton/coursera/lecture6/lec6.pdf
    real :: decay_rate = 0.9
    real :: epsilon = 1e-8
    real, allocatable, private :: rms_gradient(:)
  contains
    procedure :: init => init_rmsprop
    procedure :: minimize => minimize_rmsprop
  end type rmsprop

  type, extends(optimizer_base_type) :: adam
    !! Adam optimizer by Kingma and Ba (2014), with optional decoupled weight
    !! decay regularization (AdamW) by Loshchilov and Hutter (2017).
    !!
    !! Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic
    !! optimization. arXiv preprint arXiv:1412.6980.
    !! https://arxiv.org/abs/1412.6980
    !!
    !! Loshchilov, I. and Hutter, F., 2017. Decoupled weight decay
    !! regularization. arXiv preprint arXiv:1711.05101.
    !! https://arxiv.org/abs/1711.05101
    real :: beta1 = 0.9
    real :: beta2 = 0.999
    real :: epsilon = 1e-8
    real :: weight_decay_l2 = 0  ! L2 regularization (Adam)
    real :: weight_decay_decoupled = 0 ! decoupled weight decay regularization (AdamW)
    real, allocatable, private :: m(:), v(:)
    integer, private :: t = 0
  contains
    procedure :: init => init_adam
    procedure :: minimize => minimize_adam
  end type adam

contains

  impure elemental subroutine init_sgd(self, num_params)
    class(sgd), intent(inout) :: self
    integer, intent(in) :: num_params
    if (self % momentum > 0 .and. .not. allocated(self % velocity)) then
      allocate(self % velocity(num_params))
      self % velocity = 0
    end if
  end subroutine init_sgd


  pure subroutine minimize_sgd(self, param, gradient)
    !! Concrete implementation of a stochastic gradient descent optimizer
    !! update rule.
    class(sgd), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)

    if (self % momentum > 0) then
      ! Apply momentum update
      self % velocity = self % momentum * self % velocity &
        - self % learning_rate * gradient
      if (self % nesterov) then
        ! Apply Nesterov update
        param = param + self % momentum * self % velocity &
          - self % learning_rate * gradient
      else
        param = param + self % velocity
      end if
    else
      ! Apply regular update
      param = param - self % learning_rate * gradient
    end if

  end subroutine minimize_sgd


  impure elemental subroutine init_rmsprop(self, num_params)
    class(rmsprop), intent(inout) :: self
    integer, intent(in) :: num_params
    if (.not. allocated(self % rms_gradient)) then
      allocate(self % rms_gradient(num_params))
      self % rms_gradient = 0
    end if
  end subroutine init_rmsprop


  pure subroutine minimize_rmsprop(self, param, gradient)
    !! Concrete implementation of a RMSProp optimizer update rule.
    class(rmsprop), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)

    ! Compute the RMS of the gradient using the RMSProp rule
    self % rms_gradient = self % decay_rate * self % rms_gradient &
      + (1 - self % decay_rate) * gradient**2

    ! Update the network parameters based on the new RMS of the gradient
    param = param - self % learning_rate &
      / sqrt(self % rms_gradient + self % epsilon) * gradient

  end subroutine minimize_rmsprop


  impure elemental subroutine init_adam(self, num_params)
    class(adam), intent(inout) :: self
    integer, intent(in) :: num_params
    if (.not. allocated(self % m)) then
      allocate(self % m(num_params), self % v(num_params))
      self % m = 0
      self % v = 0
    end if
  end subroutine init_adam


  pure subroutine minimize_adam(self, param, gradient)
    !! Concrete implementation of an Adam optimizer update rule.
    class(adam), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)

    self % t = self % t + 1

    ! If weight_decay_l2 > 0, use L2 regularization;
    ! otherwise, default to regular Adam.
    associate(g => gradient + self % weight_decay_l2 * param)
      self % m = self % beta1 * self % m + (1 - self % beta1) * g
      self % v = self % beta2 * self % v + (1 - self % beta2) * g**2
    end associate

    ! Compute bias-corrected first and second moment estimates.
    associate( &
      m_hat => self % m / (1 - self % beta1**self % t), &
      v_hat => self % v / (1 - self % beta2**self % t) &
    )

    ! Update parameters.
    param = param - self % learning_rate &
      * m_hat / (sqrt(v_hat) + self % epsilon) - self % weight_decay_decoupled * param

    end associate

  end subroutine minimize_adam

end module nf_optimizers
