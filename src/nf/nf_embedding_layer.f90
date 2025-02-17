module nf_embedding_layer

  use nf_activation, only: activation_function
  use nf_base_layer, only: base_layer

  implicit none

  private
  public :: embedding_layer

  type, extends(base_layer) :: embedding_layer
    !! Embedding Layer
    !! Stores inputs as a trainable lookup table. Inputs are
    !! integer indicies in a dictionary of `vocab_size`.
    !! This layer converts them into a table of shape
    !! (`sequence_length`, `model_dimension`)
    integer :: sequence_length, vocab_size, model_dimension

    real, allocatable :: weights(:, :)
    real, allocatable :: output(:, :)
    real, allocatable :: gradient(:, :) ! input gradient
    real, allocatable :: dw(:, :) ! weight gradients

  contains

    procedure :: backward
    procedure :: forward
    procedure :: init
    procedure :: get_num_params
    procedure :: get_params
    procedure :: get_gradients
    procedure :: set_params

  end type embedding_layer

  interface embedding_layer
    module function embedding_layer_cons(vocab_size, model_dimension) result(res)
      integer, intent(in) :: vocab_size, model_dimension
      type(embedding_layer) :: res
    end function embedding_layer_cons
  end interface embedding_layer

  interface
    pure module subroutine forward(self, input)
      !! Get vectors by indicis in the dictionary
      class(embedding_layer), intent(in out) :: self
      integer, intent(in) :: input(:)
    end subroutine forward

    pure module subroutine backward(self, input, gradient)
      !! Update gradient at `input` indices
      !! dw_i = W_i + d_output_i
      class(embedding_layer), intent(in out) :: self
      integer, intent(in) :: input(:)
      real, intent(in) :: gradient(:, :)
    end subroutine backward

    module subroutine init(self, input_shape)
      class(embedding_layer), intent(in out) :: self
      integer, intent(in) :: input_shape(:)
    end subroutine init

    pure module function get_num_params(self) result(num_params)
       class(embedding_layer), intent(in) :: self
       integer :: num_params
    end function get_num_params

    module function get_params(self) result(params)
      class(embedding_layer), intent(in), target :: self
      real, allocatable :: params(:)
    end function get_params

    module function get_gradients(self) result(gradients)
      class(embedding_layer), intent(in), target :: self
      real, allocatable :: gradients(:)
    end function get_gradients

    module subroutine set_params(self, params)
      class(embedding_layer), intent(in out) :: self
      real, intent(in), target :: params(:)
    end subroutine set_params
  end interface
end module nf_embedding_layer
