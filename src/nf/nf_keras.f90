module nf_keras

  !! This module provides procedures to read and parse Keras models
  !! from HDF5 files.

  implicit none

  private
  public :: get_keras_h5_layers, keras_layer

  type :: keras_layer

    !! Intermediate container to convey the Keras layer
    !! information to neural-fortran layer constructors.

    ! General metadata that applies to any (or most) layers
    character(:), allocatable :: class
    character(:), allocatable :: name
    character(:), allocatable :: activation

    ! Dense
    integer, allocatable :: units(:)

    ! Conv2D
    integer :: filters
    integer, allocatable :: kernel_size(:)

    ! MaxPooling2D
    integer, allocatable :: pool_size(:)
    integer, allocatable :: strides(:)

    ! Reshape
    integer, allocatable :: target_shape(:)

  end type keras_layer

  interface

    module function get_keras_h5_layers(filename) result(res)
      character(*), intent(in) :: filename
        !! HDF5 file name
      type(keras_layer), allocatable :: res(:)
    end function get_keras_h5_layers

  end interface

end module nf_keras
