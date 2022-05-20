submodule(nf_base_layer) nf_base_layer_submodule

  use nf_activation, only: activation_function, &
                           elu, elu_prime, &
                           exponential, &
                           gaussian, gaussian_prime, &
                           relu, relu_prime, &
                           sigmoid, sigmoid_prime, &
                           softplus, softplus_prime, &
                           step, step_prime, &
                           tanhf, tanh_prime

  implicit none

contains
  
  elemental module subroutine set_activation(self, activation)
    class(base_layer), intent(in out) :: self
    character(*), intent(in) :: activation

    select case(trim(activation))

      ! TODO need to figure out how to handle the alpha param
      !case('elu')
      !  self % activation => elu
      !  self % activation_prime => elu_prime
      !  self % activation_name = 'elu'

      case('exponential')
        self % activation => exponential
        self % activation_prime => exponential
        self % activation_name = 'exponential'

      case('gaussian')
        self % activation => gaussian
        self % activation_prime => gaussian_prime
        self % activation_name = 'gaussian'

      case('relu')
        self % activation => relu
        self % activation_prime => relu_prime
        self % activation_name = 'relu'

      case('sigmoid')
        self % activation => sigmoid
        self % activation_prime => sigmoid_prime
        self % activation_name = 'sigmoid'

      case('softplus')
        self % activation => softplus
        self % activation_prime => softplus_prime
        self % activation_name = 'softplus'

      case('step')
        self % activation => step
        self % activation_prime => step_prime
        self % activation_name = 'step'

      case('tanh')
        self % activation => tanhf
        self % activation_prime => tanh_prime
        self % activation_name = 'tanh'

      case default
        error stop 'Activation must be one of: ' // &
          '"elu", "exponential", "gaussian", "relu", ' // &
          '"sigmoid", "softplus", "step", or "tanh".'

    end select

  end subroutine set_activation

end submodule nf_base_layer_submodule
