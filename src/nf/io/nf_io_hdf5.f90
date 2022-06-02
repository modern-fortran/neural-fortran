module nf_io_hdf5

  !! This module provides convenience functions to read HDF5 files.

  implicit none

  private
  public :: get_hdf5_attribute_string

  interface

    module function get_hdf5_attribute_string( &
      filename, object_name, attribute_name) result(res)
      character(*), intent(in) :: filename
        !! HDF5 file name
      character(*), intent(in) :: object_name
        !! Object (group, dataset) name
      character(*), intent(in) :: attribute_name
        !! Name of the attribute to read
      character(:), allocatable :: res
    end function get_hdf5_attribute_string

  end interface

end module nf_io_hdf5
