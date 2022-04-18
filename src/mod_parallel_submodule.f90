submodule(mod_parallel) mod_parallel_submodule

  use mod_kinds, only: ik, rk
  implicit none

contains

  pure module function tile_indices(dims)
    integer(ik), intent(in) :: dims
    integer(ik) :: tile_indices(2)
    integer(ik) :: offset, tile_size

    tile_size = dims / num_images()

    !! start and end indices assuming equal tile sizes
    tile_indices(1) = (this_image() - 1) * tile_size + 1
    tile_indices(2) = tile_indices(1) + tile_size - 1

    !! if we have any remainder, distribute it to the tiles at the end
    offset = num_images() - mod(dims, num_images())
    if (this_image() > offset) then
      tile_indices(1) = tile_indices(1) + this_image() - offset - 1
      tile_indices(2) = tile_indices(2) + this_image() - offset
    end if

  end function tile_indices

end submodule mod_parallel_submodule
