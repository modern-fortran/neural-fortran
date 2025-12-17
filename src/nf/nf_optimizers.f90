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
  public :: optimizer_base_type, sgd, rmsprop, adam, adagrad

  type, abstract :: optimizer_base_type
    real :: learning_rate = 0.01
  contains
    procedure :: get_name
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
    integer, private :: start_index = 1
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
    integer, private :: start_index = 1
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
    integer, private :: start_index = 1
  contains
    procedure :: init => init_adam
    procedure :: minimize => minimize_adam
  end type adam

  type, extends(optimizer_base_type) :: adagrad
    !! Adagrad optimizer by Duchi et al. (2011)
    !!
    !! Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient
    !! methods for online learning and stochastic optimization. Journal
    !! of Machine Learning Research, 12(Jul), pp.2121-2159.
    !! http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf
    real :: epsilon = 1e-8
    real :: weight_decay_l2 = 0
    real :: learning_rate_decay = 0
    real, allocatable, private :: sum_squared_gradient(:)
    integer, private :: t = 0
    integer, private :: start_index = 1
  contains
    procedure :: init => init_adagrad
    procedure :: minimize => minimize_adagrad
  end type adagrad

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
    real, intent(in) :: gradient(:) ! Always the same size as param
    integer :: end_index

    if (self % momentum > 0) then

      ! end_index is part of the bookkeeping for updating velocity because each
      ! batch update makes two calls to minimize, one for the weights and one for
      ! the biases.
      ! We use start_index and end_index to update the appropriate sections
      ! of the velocity array.
      end_index = self % start_index + size(param) - 1

      ! Apply momentum update
      self % velocity(self % start_index:end_index) = &
        self % momentum * self % velocity(self % start_index:end_index) &
        - self % learning_rate * gradient
      if (self % nesterov) then
        ! Apply Nesterov update
        param = param + self % momentum * self % velocity(self % start_index:end_index) &
          - self % learning_rate * gradient
      else
        param = param + self % velocity(self % start_index:end_index)
      end if

      if (end_index < size(param)) then
        ! We updated the weights part, now we shift forward for the biases part
        self % start_index = end_index + 1
      else
        ! We updated the biases part, now we shift back to start for the next batch
        self % start_index = 1
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
    integer :: end_index

    end_index = self % start_index + size(param) - 1

    ! Compute the RMS of the gradient using the RMSProp rule
    self % rms_gradient(self % start_index:end_index) = &
      self % decay_rate * self % rms_gradient(self % start_index:end_index) &
      + (1 - self % decay_rate) * gradient**2

    ! Update the network parameters based on the new RMS of the gradient
    param = param - self % learning_rate &
      / sqrt(self % rms_gradient(self % start_index:end_index) + self % epsilon) &
      * gradient

    if (end_index < size(param)) then
      ! We updated the weights part, now we shift forward for the biases part
      self % start_index = end_index + 1
    else
      ! We updated the biases part, now we shift back to start for the next batch
      self % start_index = 1
    end if

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
    integer :: end_index

    end_index = self % start_index + size(param) - 1

    self % t = self % t + 1

    ! If weight_decay_l2 > 0, use L2 regularization;
    ! otherwise, default to regular Adam.
    associate(g => gradient + self % weight_decay_l2 * param)
      self % m(self % start_index:end_index) = &
        self % beta1 * self % m(self % start_index:end_index) &
        + (1 - self % beta1) * g
      self % v(self % start_index:end_index) = &
        self % beta2 * self % v(self % start_index:end_index) &
        + (1 - self % beta2) * g**2
    end associate

    ! Compute bias-corrected first and second moment estimates.
    associate( &
      m_hat => self % m(self % start_index:end_index) / (1 - self % beta1**self % t), &
      v_hat => self % v(self % start_index:end_index) / (1 - self % beta2**self % t) &
    )

    ! Update parameters.
    param = param &
      - self % learning_rate * (m_hat / (sqrt(v_hat) + self % epsilon) &
      + self % weight_decay_decoupled * param)

    end associate

    if (end_index < size(param)) then
      ! We updated the weights part, now we shift forward for the biases part
      self % start_index = end_index + 1
    else
      ! We updated the biases part, now we shift back to start for the next batch
      self % start_index = 1
    end if

  end subroutine minimize_adam


  impure elemental subroutine init_adagrad(self, num_params)
    class(adagrad), intent(inout) :: self
    integer, intent(in) :: num_params
    if (.not. allocated(self % sum_squared_gradient)) then
      allocate(self % sum_squared_gradient(num_params))
      self % sum_squared_gradient = 0
    end if
  end subroutine init_adagrad


  pure subroutine minimize_adagrad(self, param, gradient)
    !! Concrete implementation of an Adagrad optimizer update rule.
    class(adagrad), intent(inout) :: self
    real, intent(inout) :: param(:)
    real, intent(in) :: gradient(:)
    integer :: end_index

    end_index = self % start_index + size(param) - 1

    ! Update the current time step
    self % t = self % t + 1

    associate( &
      ! If weight_decay_l2 > 0, use L2 regularization;
      ! otherwise, default to regular Adagrad.
      g => gradient + self % weight_decay_l2 * param, &
      ! Amortize the learning rate as function of the current time step.
      learning_rate => self % learning_rate &
        / (1 + (self % t - 1) * self % learning_rate_decay) &
    )

      self % sum_squared_gradient(self % start_index:end_index) = &
        self % sum_squared_gradient(self % start_index:end_index) + g**2

      param = param - learning_rate * g &
        / (sqrt(self % sum_squared_gradient(self % start_index:end_index)) &
        + self % epsilon)

    end associate

    if (end_index < size(param)) then
      ! We updated the weights part, now we shift forward for the biases part
      self % start_index = end_index + 1
    else
      ! We updated the biases part, now we shift back to start for the next batch
      self % start_index = 1
    end if

  end subroutine minimize_adagrad


  ! Utility Functions
  !! Returns the default optimizer corresponding to the provided name
  function get_optimizer_by_name(optimizer_name) result(res)
    character(len=*), intent(in) :: optimizer_name
    class(optimizer_base_type), allocatable :: res

    select case(trim(optimizer_name))
    case('adagrad')
      allocate ( res, source = adagrad() )

    case('adam')
      allocate ( res, source = adam() )

    case('rmsprop')
      allocate ( res, source = rmsprop() )

    case('sgd')
     allocate ( res, source = sgd() )

    case default
        error stop 'optimizer_name must be one of: ' // &
          '"adagrad", "adam", "rmsprop", "sgd".'
    end select

  end function get_optimizer_by_name


  !! Returns the name of the optimizer
  pure function get_name(self) result(name)
    class(optimizer_base_type), intent(in) :: self
    character(:), allocatable :: name

    select type (self)
    class is (adagrad)
      name = 'adagrad'
    class is (adam)
      name = 'adam'
    class is (rmsprop)
      name = 'rmsprop'
    class is (sgd)
      name = 'sgd'
    class default
      error stop 'Unknown optimizer type.'
    end select

  end function get_name

end module nf_optimizers
