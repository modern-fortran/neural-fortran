submodule(nf_io_binary) nf_io_binary_submodule

  implicit none

  integer, parameter :: message_len = 128

contains

  module subroutine read_binary_file_1d(filename, dtype, nrec, array)
    character(*), intent(in) :: filename
    integer, intent(in) :: dtype, nrec
    real, allocatable, intent(in out) :: array(:)
    integer :: fileunit
    character(message_len) :: io_message
    integer :: io_status
    io_status = 0
    open(newunit=fileunit, file=filename, access='direct', action='read', &
      recl=dtype * nrec, status='old', iostat=io_status, iomsg=io_message)
    if (io_status /= 0) error stop trim(io_message)
    allocate(array(nrec))
    read(fileunit, rec=1) array
    close(fileunit)
  end subroutine read_binary_file_1d

  module subroutine read_binary_file_2d(filename, dtype, dsize, nrec, array)
    character(*), intent(in) :: filename
    integer, intent(in) :: dtype, dsize, nrec
    real, allocatable, intent(in out) :: array(:,:)
    integer :: fileunit, i
    character(message_len) :: io_message
    integer :: io_status
    io_status = 0
    open(newunit=fileunit, file=filename, access='direct', action='read', &
      recl=dtype * dsize, status='old', iostat=io_status, iomsg=io_message)
    if (io_status /= 0) error stop trim(io_message)
    allocate(array(dsize, nrec))
    do i = 1, nrec
      read(fileunit, rec=i) array(:,i)
    end do
    close(fileunit)
  end subroutine read_binary_file_2d

  module subroutine read_cifar10(filename, nrec, images, labels)
    character(*), intent(in) :: filename
    integer, intent(in) :: nrec

    real, allocatable, intent(out) :: images(:,:)
    real, allocatable, intent(out) :: labels(:)

    call read_cifar_common(filename, nrec, images, labels, .false.)
  end subroutine read_cifar10

  module subroutine read_cifar100(filename, nrec, images, labels)
    character(*), intent(in) :: filename
    integer, intent(in) :: nrec

    real, allocatable, intent(out) :: images(:,:)
    real, allocatable, intent(out) :: labels(:)

    call read_cifar_common(filename, nrec, images, labels, .true.)
  end subroutine read_cifar100

  subroutine read_cifar_common(filename, nrec, images, labels, cifar_100)
    character(*), intent(in) :: filename
    integer, intent(in) :: nrec
    logical, intent(in) :: cifar_100

    real, allocatable, intent(out) :: images(:,:)
    real, allocatable, intent(out) :: labels(:)

    integer(1), allocatable :: buffer(:,:)
    integer :: unit, ios, i, j, val
    character(len=256) :: msg

    integer :: record_size, label_offset

    ! CIFAR-10: 1 label
    ! CIFAR-100: 2 labels (coarse + fine)
    record_size = 3072 + merge(2, 1, cifar_100)

    allocate(buffer(record_size, nrec))

    open(newunit=unit, file=filename, access='stream', &
         form='unformatted', status='old', action='read', &
         iostat=ios, iomsg=msg)

    if (ios /= 0) error stop trim(msg)

    read(unit) buffer
    close(unit)

    allocate(images(3072, nrec))
    allocate(labels(nrec))

    do i = 1, nrec
      if (cifar_100) then
        ! Choose which label you want:
        label_offset = 2   ! fine label (more commonly used in ML)
        ! label_offset = 1  ! coarse label (alternative)
      else
        label_offset = 1
      end if

      val = buffer(label_offset, i)
      if (val < 0) val = val + 256
      labels(i) = real(val)

      ! pixel data starts after both labels in CIFAR-100
      do j = 1, 3072
        val = buffer(j + merge(2, 1, cifar_100), i)
        if (val < 0) val = val + 256

        images(j, i) = real(val) / 255.0
      end do

    end do

  end subroutine read_cifar_common

end submodule nf_io_binary_submodule
