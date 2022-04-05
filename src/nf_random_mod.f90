module nf_random_mod

  ! Provides a random number generator with
  ! normal distribution, centered on zero.

  use mod_kinds, only: ik, rk

  implicit none

  private
  public :: randn

  interface randn

    module function randn1d(n) result(r)
      ! Generates n random numbers with a normal distribution.
      implicit none
      integer(ik), intent(in) :: n
      real(rk) :: r(n)
    end function randn1d

    module function randn2d(m, n) result(r)
      ! Generates m x n random numbers with a normal distribution.
      integer(ik), intent(in) :: m, n
      real(rk) :: r(m, n)
    end function randn2d
  
  end interface randn

end module nf_random_mod
