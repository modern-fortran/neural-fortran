module nf_io_hdf5

  !! This module provides convenience functions to read HDF5 files.

  use iso_fortran_env, only: real32

  implicit none

  private
  public :: get_hdf5_dataset, hdf5_attribute_string

  interface
    module function hdf5_attribute_string( &
      filename, object_name, attribute_name) result(res)
      !! Read and return an HDF5 variable-length UTF-8 string attribute.
      character(*), intent(in) :: filename
        !! HDF5 file name
      character(*), intent(in) :: object_name
        !! Object (group, dataset) name
      character(*), intent(in) :: attribute_name
        !! Name of the attribute to read
      character(:), allocatable :: res
    end function hdf5_attribute_string
  end interface

  interface get_hdf5_dataset

    module subroutine get_hdf5_dataset_real32_1d(filename, object_name, values)
      !! Read a 1-d real32 array from an HDF5 dataset.
      character(*), intent(in) :: filename
        !! HDF5 file name
      character(*), intent(in) :: object_name
        !! Object (dataset) name
      real(real32), allocatable, intent(out) :: values(:)
        !! Array to store the dataset values into
    end subroutine get_hdf5_dataset_real32_1d

    module subroutine get_hdf5_dataset_real32_2d(filename, object_name, values)
      !! Read a 2-d real32 array from an HDF5 dataset.
      character(*), intent(in) :: filename
        !! HDF5 file name
      character(*), intent(in) :: object_name
        !! Object (dataset) name
      real(real32), allocatable, intent(out) :: values(:,:)
        !! Array to store the dataset values into
    end subroutine get_hdf5_dataset_real32_2d
  
    module subroutine get_hdf5_dataset_real32_4d(filename, object_name, values)
      !! Read a 4-d real32 array from an HDF5 dataset.
      character(*), intent(in) :: filename
        !! HDF5 file name
      character(*), intent(in) :: object_name
        !! Object (dataset) name
      real(real32), allocatable, intent(out) :: values(:,:,:,:)
        !! Array to store the dataset values into
    end subroutine get_hdf5_dataset_real32_4d

  end interface get_hdf5_dataset

end module nf_io_hdf5
