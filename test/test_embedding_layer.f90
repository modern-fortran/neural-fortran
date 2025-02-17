program test_embedding_layer
  use iso_fortran_env, only: stderr => error_unit
  use nf_embedding_layer, only: embedding_layer
  implicit none

  logical :: ok = .true.
  integer :: sample_input(3) = [2, 1, 3]
  type(embedding_layer) :: embedding

  embedding = embedding_layer(sequence_length=3, vocab_size=4, model_dimension=2)
  call embedding % init([0])
  embedding % weights = reshape([0.1, 0.3, 0.5, 0.7, 0.2, 0.4, 0.6, 0.8], [4, 2])
  call embedding % forward(sample_input)
end program test_embedding_layer