submodule(mod_network) mod_network_submodule

  use mod_kinds, only: ik, rk
  use mod_layer, only: db_init, dw_init, db_co_sum, dw_co_sum
  use mod_parallel, only: tile_indices

  implicit none

contains

  module function net_constructor(dims, activation) result(net)
    integer(ik), intent(in) :: dims(:)
    character(len=*), intent(in), optional :: activation
    type(network_type) :: net
    call net % init(dims)
    if (present(activation)) then
      call net % set_activation(activation)
    else
      call net % set_activation('sigmoid')
    end if
    call net % sync(1)
  end function net_constructor

  pure real(rk) module function accuracy(self, x, y)
    class(network_type), intent(in) :: self
    real(rk), intent(in) :: x(:,:), y(:,:)
    integer(ik) :: i, good
    good = 0
    do i = 1, size(x, dim=2)
      if (all(maxloc(self % output(x(:,i))) == maxloc(y(:,i)))) then
        good = good + 1
      end if
    end do
    accuracy = real(good, kind=rk) / size(x, dim=2)
  end function accuracy


  pure module subroutine backprop(self, y, dw, db)
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: y(:)
    type(array2d), allocatable, intent(out) :: dw(:)
    type(array1d), allocatable, intent(out) :: db(:)
    integer(ik) :: n, nm

    associate(dims => self % dims, layers => self % layers)

      call db_init(db, dims)
      call dw_init(dw, dims)

      n = size(dims)
      db(n) % array = (layers(n) % a - y) * self % layers(n) % activation_prime(layers(n) % z)
      dw(n-1) % array = matmul(reshape(layers(n-1) % a, [dims(n-1), 1]),&
                               reshape(db(n) % array, [1, dims(n)]))

      do n = size(dims) - 1, 2, -1
        db(n) % array = matmul(layers(n) % w, db(n+1) % array)&
                      * self % layers(n) % activation_prime(layers(n) % z)
        dw(n-1) % array = matmul(reshape(layers(n-1) % a, [dims(n-1), 1]),&
                                 reshape(db(n) % array, [1, dims(n)]))
      end do

    end associate

  end subroutine backprop


  pure module subroutine fwdprop(self, x)
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:)
    integer(ik) :: n
    associate(layers => self % layers)
      layers(1) % a = x
      do n = 2, size(layers)
        layers(n) % z = matmul(transpose(layers(n-1) % w), layers(n-1) % a) + layers(n) % b
        layers(n) % a = self % layers(n) % activation(layers(n) % z)
      end do
    end associate
  end subroutine fwdprop


  module subroutine init(self, dims)
    class(network_type), intent(in out) :: self
    integer(ik), intent(in) :: dims(:)
    integer(ik) :: n
    self % dims = dims
    if (.not. allocated(self % layers)) allocate(self % layers(size(dims)))
    do n = 1, size(dims) - 1
      self % layers(n) = layer_type(dims(n), dims(n+1))
    end do
    self % layers(n) = layer_type(dims(n), 1)
    self % layers(1) % b = 0
    self % layers(size(dims)) % w = 0
  end subroutine init


  module subroutine load(self, filename)
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer(ik) :: fileunit, n, num_layers, layer_idx
    integer(ik), allocatable :: dims(:)
    character(len=100) :: buffer !! activation string
    open(newunit=fileunit, file=filename, status='old', action='read')
    read(fileunit, *) num_layers
    allocate(dims(num_layers))
    read(fileunit, *) dims
    call self % init(dims)
    do n = 1, num_layers
      read(fileunit, *) layer_idx, buffer
      call self % layers(layer_idx) % set_activation(trim(buffer))
    end do
    do n = 2, size(self % dims)
      read(fileunit, *) self % layers(n) % b
    end do
    do n = 1, size(self % dims) - 1
      read(fileunit, *) self % layers(n) % w
    end do
    close(fileunit)
  end subroutine load


  pure real(rk) module function loss(self, x, y)
    class(network_type), intent(in) :: self
    real(rk), intent(in) :: x(:), y(:)
    loss = 0.5 * sum((y - self % output(x))**2) / size(x)
  end function loss


  pure module function output_single(self, x) result(a)
    class(network_type), intent(in) :: self
    real(rk), intent(in) :: x(:)
    real(rk), allocatable :: a(:)
    integer(ik) :: n
    associate(layers => self % layers)
      a = self % layers(2) % activation(matmul(transpose(layers(1) % w), x) + layers(2) % b)
      do n = 3, size(layers)
        a = self % layers(n) % activation(matmul(transpose(layers(n-1) % w), a) + layers(n) % b)
      end do
    end associate
  end function output_single


  pure module function output_batch(self, x) result(a)
    class(network_type), intent(in) :: self
    real(rk), intent(in) :: x(:,:)
    real(rk), allocatable :: a(:,:)
    integer(ik) :: i
    allocate(a(self % dims(size(self % dims)), size(x, dim=2)))
    do i = 1, size(x, dim=2)
     a(:,i) = self % output_single(x(:,i))
    end do
  end function output_batch


  module subroutine save(self, filename)
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: filename
    integer(ik) :: fileunit, n
    open(newunit=fileunit, file=filename)
    write(fileunit, fmt=*) size(self % dims)
    write(fileunit, fmt=*) self % dims
    do n = 1, size(self % dims)
      write(fileunit, fmt=*) n, self % layers(n) % activation_str
    end do
    do n = 2, size(self % dims)
      write(fileunit, fmt=*) self % layers(n) % b
    end do
    do n = 1, size(self % dims) - 1
      write(fileunit, fmt=*) self % layers(n) % w
    end do
    close(fileunit)
  end subroutine save


  pure module subroutine set_activation_equal(self, activation)
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: activation
    call self % layers(:) % set_activation(activation)
  end subroutine set_activation_equal


  pure module subroutine set_activation_layers(self, activation)
    class(network_type), intent(in out) :: self
    character(len=*), intent(in) :: activation(size(self % layers))
    call self % layers(:) % set_activation(activation)
  end subroutine set_activation_layers

  module subroutine sync(self, image)
    class(network_type), intent(in out) :: self
    integer(ik), intent(in) :: image
    integer(ik) :: n
    if (num_images() == 1) return
    layers: do n = 1, size(self % dims)
      call co_broadcast(self % layers(n) % b, image)
      call co_broadcast(self % layers(n) % w, image)
    end do layers
  end subroutine sync

  module subroutine train_batch(self, x, y, eta)
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:,:), y(:,:), eta
    type(array1d), allocatable :: db(:), db_batch(:)
    type(array2d), allocatable :: dw(:), dw_batch(:)
    integer(ik) :: i, im, n, nm
    integer(ik) :: is, ie, indices(2)

    im = size(x, dim=2) ! mini-batch size
    nm = size(self % dims) ! number of layers

    ! get start and end index for mini-batch
    indices = tile_indices(im)
    is = indices(1)
    ie = indices(2)

    call db_init(db_batch, self % dims)
    call dw_init(dw_batch, self % dims)

    do concurrent(i = is:ie)
      call self % fwdprop(x(:,i))
      call self % backprop(y(:,i), dw, db)
      do concurrent(n = 1:nm)
        dw_batch(n) % array =  dw_batch(n) % array + dw(n) % array
        db_batch(n) % array =  db_batch(n) % array + db(n) % array
      end do
    end do

    if (num_images() > 1) then
      call dw_co_sum(dw_batch)
      call db_co_sum(db_batch)
    end if

    call self % update(dw_batch, db_batch, eta / im)

  end subroutine train_batch

  module subroutine train_epochs(self, x, y, eta, num_epochs, batch_size)
    class(network_type), intent(in out) :: self
    integer(ik), intent(in) :: num_epochs, batch_size
    real(rk), intent(in) :: x(:,:), y(:,:), eta

    integer(ik) :: i, n, nsamples, nbatch
    integer(ik) :: batch_start, batch_end

    real(rk) :: pos

    nsamples = size(y, dim=2)
    nbatch = nsamples / batch_size

    epochs: do n = 1, num_epochs
      batches: do i = 1, nbatch
      
        !pull a random mini-batch from the dataset  
        call random_number(pos)
        batch_start = int(pos * (nsamples - batch_size + 1))
        if (batch_start == 0) batch_start = 1
        batch_end = batch_start + batch_size - 1
   
        call self % train(x(:,batch_start:batch_end), y(:,batch_start:batch_end), eta)
       
      end do batches
    end do epochs

  end subroutine train_epochs


  pure module subroutine train_single(self, x, y, eta)
    class(network_type), intent(in out) :: self
    real(rk), intent(in) :: x(:), y(:), eta
    type(array2d), allocatable :: dw(:)
    type(array1d), allocatable :: db(:)
    call self % fwdprop(x)
    call self % backprop(y, dw, db)
    call self % update(dw, db, eta)
  end subroutine train_single


  pure module subroutine update(self, dw, db, eta)
    class(network_type), intent(in out) :: self
    class(array2d), intent(in) :: dw(:)
    class(array1d), intent(in) :: db(:)
    real(rk), intent(in) :: eta
    integer(ik) :: n

    associate(layers => self % layers, nm => size(self % dims))
      ! update biases
      do concurrent(n = 2:nm)
        layers(n) % b = layers(n) % b - eta * db(n) % array
      end do
      ! update weights
      do concurrent(n = 1:nm-1)
        layers(n) % w = layers(n) % w - eta * dw(n) % array
      end do
    end associate

  end subroutine update

end submodule mod_network_submodule
