submodule(nf_io_hdf5) nf_io_hdf5_submodule

  use iso_fortran_env, only: int64, real32, stderr => error_unit
  use h5fortran, only: hdf5_file
  use hdf5, only: H5F_ACC_RDONLY_F, HID_T, &
                  h5aget_type_f, h5aopen_by_name_f, h5aread_f, &
                  h5fclose_f, h5fopen_f
  use iso_c_binding, only: c_char, c_f_pointer, c_loc, c_null_char, c_ptr

  implicit none

contains

  module function hdf5_attribute_string( &
    filename, object_name, attribute_name) result(res)

    character(*), intent(in) :: filename
    character(*), intent(in) :: object_name
    character(*), intent(in) :: attribute_name
    character(:), allocatable :: res

    ! Make sufficiently large to hold most attributes
    integer, parameter :: BUFLEN = 10000

    type(c_ptr) :: f_ptr
    type(c_ptr), target :: buffer
    character(len=BUFLEN, kind=c_char), pointer :: string => null()
    integer(HID_T) :: fid, aid, atype
    integer :: hdferr

    ! Open the file and get the type of the attribute
    call h5fopen_f(filename, H5F_ACC_RDONLY_F, fid, hdferr)
    call h5aopen_by_name_f(fid, object_name, attribute_name, aid, hdferr)
    call h5aget_type_f(aid, atype, hdferr)

    ! Read the data
    f_ptr = c_loc(buffer)
    call h5aread_f(aid, atype, f_ptr, hdferr)
    call c_f_pointer(buffer, string)

    ! Close the file 
    call h5fclose_f(fid, hdferr)
    
    res = string(:index(string, c_null_char) - 1)

  end function hdf5_attribute_string


  module subroutine get_hdf5_dataset_real32_1d(filename, object_name, values)

    character(*), intent(in) :: filename
    character(*), intent(in) :: object_name
    real(real32), allocatable, intent(out) :: values(:)

    type(hdf5_file) :: f
    integer(int64), allocatable :: dims(:)

    call f % open(filename, 'r')
    call f % shape(object_name, dims)

    allocate(values(dims(1)))

    call f % read(object_name, values)
    call f % close()

  end subroutine get_hdf5_dataset_real32_1d


  module subroutine get_hdf5_dataset_real32_2d(filename, object_name, values)

    character(*), intent(in) :: filename
    character(*), intent(in) :: object_name
    real(real32), allocatable, intent(out) :: values(:,:)

    type(hdf5_file) :: f
    integer(int64), allocatable :: dims(:)

    call f % open(filename, 'r')
    call f % shape(object_name, dims)

    allocate(values(dims(1), dims(2)))

    call f % read(object_name, values)
    call f % close()

    ! Transpose the array to respect Keras's storage order
    values = transpose(values)

  end subroutine get_hdf5_dataset_real32_2d


  module subroutine get_hdf5_dataset_real32_4d(filename, object_name, values)

    character(*), intent(in) :: filename
    character(*), intent(in) :: object_name
    real(real32), allocatable, intent(out) :: values(:,:,:,:)

    type(hdf5_file) :: f
    integer(int64), allocatable :: dims(:)

    call f % open(filename, 'r')
    call f % shape(object_name, dims)

    allocate(values(dims(1), dims(2), dims(3), dims(4)))

    call f % read(object_name, values)
    call f % close()

  end subroutine get_hdf5_dataset_real32_4d

end submodule nf_io_hdf5_submodule
