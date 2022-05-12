module nf_random

  !! Provides a random number generator with
  !! normal distribution, centered on zero.

  implicit none

  private
  public :: randn

  interface randn

    module function randn_1d(i) result(r)
      !! Generates i random numbers with a normal distribution,
      !! using the Box-Muller method.
      implicit none
      integer, intent(in) :: i
      real :: r(i)
    end function randn_1d

    module function randn_2d(i, j) result(r)
      !! Generates i x j random numbers with a normal distribution,
      !! using the Box-Muller method.
      implicit none
      integer, intent(in) :: i, j
      real :: r(i,j)
    end function randn_2d

    module function randn_4d(i, j, k, l) result(r)
      !! Generates i x j x k x l random numbers with a normal distribution,
      !! using the Box-Muller method.
      implicit none
      integer, intent(in) :: i, j, k, l
      real :: r(i,j,k,l)
    end function randn_4d

  end interface randn

end module nf_random
