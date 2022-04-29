module nf_parallel

  implicit none

  private
  public :: tile_indices

  interface
  
    pure module function tile_indices(dims) result(res)
      !! Given input global array size, return start and end index
      !! of a parallel 1-d tile that correspond to this image.
      implicit none
      integer, intent(in) :: dims
      integer :: res(2)
    end function tile_indices
  
  end interface

end module nf_parallel
