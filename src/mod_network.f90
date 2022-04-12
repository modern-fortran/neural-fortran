module mod_network

  use mod_kinds, only: ik, rk
  use mod_layer, only: array1d, array2d, layer_type

  implicit none

  private
  public :: network_type

  type :: network_type

    type(layer_type), allocatable :: layers(:)
    integer(ik), allocatable :: dims(:)

  contains

    procedure, public, pass(self) :: accuracy
    procedure, public, pass(self) :: backprop
    procedure, public, pass(self) :: fwdprop
    procedure, public, pass(self) :: init
    procedure, public, pass(self) :: load
    procedure, public, pass(self) :: loss
    procedure, public, pass(self) :: output_batch
    procedure, public, pass(self) :: output_single
    procedure, public, pass(self) :: save
    procedure, public, pass(self) :: set_activation_equal
    procedure, public, pass(self) :: set_activation_layers
    procedure, public, pass(self) :: sync
    procedure, public, pass(self) :: train_batch
    procedure, public, pass(self) :: train_epochs
    procedure, public, pass(self) :: train_single
    procedure, public, pass(self) :: update

    generic, public :: output => output_batch, output_single
    generic, public :: set_activation => set_activation_equal, set_activation_layers
    generic, public :: train => train_batch, train_epochs, train_single

  end type network_type

  interface network_type
    
    type(network_type) module function net_constructor(dims, activation) result(net)
      !! Network class constructor. Size of input array dims indicates the total
      !! number of layers (input + hidden + output), and the value of its elements
      !! corresponds the size of each layer.
      implicit none
      integer(ik), intent(in) :: dims(:)
      character(len=*), intent(in), optional :: activation
    end function net_constructor
          
  end interface network_type

  interface

    pure real(rk) module function accuracy(self, x, y)
      !! Given input x and output y, evaluates the position of the
      !! maximum value of the output and returns the number of matches
      !! relative to the size of the dataset.
      implicit none
      class(network_type), intent(in) :: self
      real(rk), intent(in) :: x(:,:), y(:,:)
    end function accuracy


    pure module subroutine backprop(self, y, dw, db)
      !! Applies a backward propagation through the network
      !! and returns the weight and bias gradients.
      implicit none
      class(network_type), intent(in out) :: self
      real(rk), intent(in) :: y(:)
      type(array2d), allocatable, intent(out) :: dw(:)
      type(array1d), allocatable, intent(out) :: db(:)
    end subroutine backprop


    pure module subroutine fwdprop(self, x)
      !! Performs the forward propagation and stores arguments to activation
      !! functions and activations themselves for use in backprop.
      implicit none
      class(network_type), intent(in out) :: self
      real(rk), intent(in) :: x(:)
    end subroutine fwdprop


    module subroutine init(self, dims)
      !! Allocates and initializes the layers with given dimensions dims.
      implicit none
      class(network_type), intent(in out) :: self
      integer(ik), intent(in) :: dims(:)
    end subroutine init


    module subroutine load(self, filename)
      !! Loads the network from file.
      implicit none
      class(network_type), intent(in out) :: self
      character(len=*), intent(in) :: filename
    end subroutine load


    pure module real(rk) function loss(self, x, y)
      !! Given input x and expected output y, returns the loss of the network.
      implicit none
      class(network_type), intent(in) :: self
      real(rk), intent(in) :: x(:), y(:)
    end function loss


    pure module function output_single(self, x) result(a)
      !! Use forward propagation to compute the output of the network.
      !! This specific procedure is for a single sample of 1-d input data.
      implicit none
      class(network_type), intent(in) :: self
      real(rk), intent(in) :: x(:)
      real(rk), allocatable :: a(:)
    end function output_single


    pure module function output_batch(self, x) result(a)
      !! Use forward propagation to compute the output of the network.
      !! This specific procedure is for a batch of 1-d input data.
      implicit none
      class(network_type), intent(in) :: self
      real(rk), intent(in) :: x(:,:)
      real(rk), allocatable :: a(:,:)
    end function output_batch

    module subroutine save(self, filename)
      !! Saves the network to a file.
      implicit none
      class(network_type), intent(in out) :: self
      character(len=*), intent(in) :: filename
    end subroutine save


    pure module subroutine set_activation_equal(self, activation)
      !! A thin wrapper around layer % set_activation().
      !! This method can be used to set an activation function
      !! for all layers at once.
      implicit none
      class(network_type), intent(in out) :: self
      character(len=*), intent(in) :: activation
    end subroutine set_activation_equal


    pure module subroutine set_activation_layers(self, activation)
      !! A thin wrapper around layer % set_activation().
      !! This method can be used to set different activation functions
      !! for each layer separately.
      implicit none
      class(network_type), intent(in out) :: self
      character(len=*), intent(in) :: activation(size(self % layers))
    end subroutine set_activation_layers

    module subroutine sync(self, image)
      !! Broadcasts network weights and biases from
      !! specified image to all others.
      implicit none
      class(network_type), intent(in out) :: self
      integer(ik), intent(in) :: image
    end subroutine sync


    module subroutine train_batch(self, x, y, eta)
      !! Trains a network using input data x and output data y,
      !! and learning rate eta. The learning rate is normalized
      !! with the size of the data batch.
      implicit none
      class(network_type), intent(in out) :: self
      real(rk), intent(in) :: x(:,:), y(:,:), eta
    end subroutine train_batch


    module subroutine train_epochs(self, x, y, eta, num_epochs, batch_size)
      !! Trains for num_epochs epochs with mini-bachtes of size equal to batch_size.
      implicit none
      class(network_type), intent(in out) :: self
      integer(ik), intent(in) :: num_epochs, batch_size
      real(rk), intent(in) :: x(:,:), y(:,:), eta
    end subroutine train_epochs


    pure module subroutine train_single(self, x, y, eta)
      !! Trains a network using a single set of input data x and output data y,
      !! and learning rate eta.
      implicit none
      class(network_type), intent(in out) :: self
      real(rk), intent(in) :: x(:), y(:), eta
    end subroutine train_single


    pure module subroutine update(self, dw, db, eta)
      !! Updates network weights and biases with gradients dw and db,
      !! scaled by learning rate eta.
      implicit none
      class(network_type), intent(in out) :: self
      class(array2d), intent(in) :: dw(:)
      class(array1d), intent(in) :: db(:)
      real(rk), intent(in) :: eta
    end subroutine update

  end interface

end module mod_network
