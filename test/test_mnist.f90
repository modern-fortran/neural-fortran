program test_mnist

  use mod_mnist, only: load_mnist
  use mod_kinds, only: ik, rk

  implicit none

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)
  real(rk), allocatable :: va_images(:,:), va_labels(:)

  print *, 'Reading MNIST data..'
  call load_mnist(tr_images, tr_labels, te_images, te_labels, va_images, va_labels)
  print *, 'Training data:'
  print *, shape(tr_images), minval(tr_images), maxval(tr_images), sum(tr_images) / size(tr_images)
  print *, shape(tr_labels), sum(tr_labels) / size(tr_labels)
  print *, 'Testing data:'
  print *, shape(te_images), minval(te_images), maxval(te_images), sum(te_images) / size(te_images)
  print *, shape(te_labels), sum(te_labels) / size(te_labels)
  print *, 'Validation data:'
  print *, shape(va_images), minval(va_images), maxval(va_images), sum(va_images) / size(va_images)
  print *, shape(va_labels), sum(va_labels) / size(va_labels)

end program test_mnist
