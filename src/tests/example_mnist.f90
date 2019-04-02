program example_mnist

  ! A training example with the MNIST dataset.
  ! Uses stochastic gradient descent and mini-batch size of 100.
  ! Can be run in serial or parallel mode without modifications.

  use mod_kinds, only: ik, rk
  use mod_mnist, only: label_digits, load_mnist
  use mod_network, only: network_type

  implicit none

  real(rk), allocatable :: tr_images(:,:), tr_labels(:)
  real(rk), allocatable :: te_images(:,:), te_labels(:)

  type(network_type) :: net

  integer(ik) :: i, n, num_epochs
  integer(ik) :: batch_size

  call load_mnist(tr_images, tr_labels, te_images, te_labels)

  net = network_type([size(tr_images,dim=1), 10, size(label_digits(te_labels),dim=1)])

  batch_size = 1000
  num_epochs = 10

  if (this_image() == 1) then
    write(*, '(a,f5.2,a)') 'Initial accuracy: ',&
      net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
  end if

  call net%fit(tr_images,label_digits(tr_labels),eta=3._rk,epochs=num_epochs,batch_size=batch_size)
   
  if (this_image() == 1) then
    write(*, '(a,f5.2,a)') 'Epochs done, Accuracy: ',&
     net % accuracy(te_images, label_digits(te_labels)) * 100, ' %'
  endif


end program example_mnist
