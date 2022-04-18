module mod_parallel

  use mod_kinds, only: ik, rk
  implicit none

  private
  public :: tile_indices

  interface
  
    pure module function tile_indices(dims)
      !! Given input global array size, return start and end index
      !! of a parallel 1-d tile that correspond to this image.
      implicit none
      integer(ik), intent(in) :: dims
      integer(ik) :: tile_indices(2)
    end function tile_indices
  
  end interface

end module mod_parallel
