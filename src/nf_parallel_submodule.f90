submodule(nf_parallel) nf_parallel_submodule
  implicit none
contains

  pure module function tile_indices(dims) result(res)
    integer, intent(in) :: dims
    integer :: res(2)
    integer :: offset, tile_size

    tile_size = dims / num_images()

    ! start and end indices assuming equal tile sizes
    res(1) = (this_image() - 1) * tile_size + 1
    res(2) = res(1) + tile_size - 1

    ! if we have any remainder, distribute it to the tiles at the end
    offset = num_images() - mod(dims, num_images())
    if (this_image() > offset) then
      res(1) = res(1) + this_image() - offset - 1
      res(2) = res(2) + this_image() - offset
    end if

  end function tile_indices

end submodule nf_parallel_submodule
