submodule(mod_layer) mod_layer_submodule
  
  use mod_random, only: randn

  implicit none

contains

  type(layer_type) module function constructor(this_size, next_size) result(layer)
    integer(ik), intent(in) :: this_size, next_size
    allocate(layer % a(this_size))
    allocate(layer % z(this_size))
    layer % a = 0
    layer % z = 0
    layer % w = randn(this_size, next_size) / this_size
    layer % b = randn(this_size)
  end function constructor

  pure type(array1d) module function array1d_constructor(length) result(a)
    integer(ik), intent(in) :: length
    allocate(a % array(length))
    a % array = 0
  end function array1d_constructor
  
  pure type(array2d) module function array2d_constructor(dims) result(a)
    integer(ik), intent(in) :: dims(2)
    allocate(a % array(dims(1), dims(2)))
    a % array = 0
  end function array2d_constructor
  
  pure module subroutine db_init(db, dims)
    type(array1d), allocatable, intent(in out) :: db(:)
    integer(ik), intent(in) :: dims(:)
    integer(ik) :: n, nm
    nm = size(dims)
    allocate(db(nm))
    do n = 1, nm - 1
      db(n) = array1d(dims(n))
    end do
    db(n) = array1d(dims(n))
  end subroutine db_init
  
  pure module subroutine dw_init(dw, dims)
    type(array2d), allocatable, intent(in out) :: dw(:)
    integer(ik), intent(in) :: dims(:)
    integer(ik) :: n, nm
    nm = size(dims)
    allocate(dw(nm))
    do n = 1, nm - 1
      dw(n) = array2d(dims(n:n+1))
    end do
    dw(n) = array2d([dims(n), 1])
  end subroutine dw_init
  
  module subroutine db_co_sum(db)
    type(array1d), allocatable, intent(in out) :: db(:)
    integer(ik) :: n
    do n = 2, size(db)
      call co_sum(db(n) % array)
    end do
  end subroutine db_co_sum
  
  module subroutine dw_co_sum(dw)
    type(array2d), allocatable, intent(in out) :: dw(:)
    integer(ik) :: n
    do n = 1, size(dw) - 1
      call co_sum(dw(n) % array)
    end do
  end subroutine dw_co_sum
  
  pure elemental module subroutine set_activation(self, activation)
    class(layer_type), intent(in out) :: self
    character(len=*), intent(in) :: activation
    select case(trim(activation))
      case('gaussian')
        self % activation => gaussian
        self % activation_prime => gaussian_prime
        self % activation_str = 'gaussian'
      case('relu')
        self % activation => relu
        self % activation_prime => relu_prime
        self % activation_str = 'relu'
      case('sigmoid')
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
        self % activation_str = 'sigmoid'
      case('step')
        self % activation => step
        self % activation_prime => step_prime
        self % activation_str = 'step'
      case('tanh')
        self % activation => tanhf
        self % activation_prime => tanh_prime
        self % activation_str = 'tanh'
      case default
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
        self % activation_str = 'sigmoid'
    end select
  end subroutine set_activation


end submodule mod_layer_submodule
