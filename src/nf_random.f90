module nf_random

  !! Provides a random number generator with
  !! normal distribution, centered on zero.

  implicit none

  private
  public :: randn

  interface randn

    module function randn1d(n) result(r)
      !! Generates n random numbers with a normal distribution,
      !! using the Box-Muller method.
      implicit none
      integer, intent(in) :: n
      real :: r(n)
    end function randn1d

    module function randn2d(m, n) result(r)
      !! Generates m x n random numbers with a normal distribution,
      !! using the Box-Muller method.
      implicit none
      integer, intent(in) :: m, n
      real :: r(m,n)
    end function randn2d

  end interface randn

end module nf_random
