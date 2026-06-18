program test_datasets_cifar
  use nf, only: label_digits_cifar100, load_cifar10, load_cifar100
  use tuff, only: test, test_result
  implicit none
  type(test_result) :: tests

  tests = test("test_datasets_cifar", [ &
    test(cifar10_loads_without_test_outputs), &
    test(cifar10_loads_with_test_outputs), &
    test(cifar100_loads_without_test_outputs), &
    test(cifar100_loads_with_test_outputs), &
    test(cifar100_label_digits_are_one_hot) &
  ])

contains

  type(test_result) function cifar10_loads_without_test_outputs() result(res)
    real, allocatable :: training_images(:,:), training_images_dummy(:,:)
    real, allocatable :: training_labels(:), training_labels_dummy(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)

    res % name = 'cifar10 loads without test outputs'

    call load_cifar10(training_images, training_images_dummy, &
      training_labels, training_labels_dummy, validation_images, &
      validation_labels)

    res % ok = all(shape(training_images) == [3072, 40000]) &
      .and. size(training_labels) == 40000 &
      .and. all(shape(validation_images) == [3072, 10000]) &
      .and. size(validation_labels) == 10000 &
      .and. valid_image_values(training_images) &
      .and. valid_image_values(validation_images) &
      .and. valid_labels(training_labels, 0., 9.) &
      .and. valid_labels(validation_labels, 0., 9.)
  end function cifar10_loads_without_test_outputs

  type(test_result) function cifar10_loads_with_test_outputs() result(res)
    real, allocatable :: training_images(:,:), training_images_dummy(:,:)
    real, allocatable :: training_labels(:), training_labels_dummy(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)
    real, allocatable :: testing_images(:,:), testing_labels(:)

    res % name = 'cifar10 loads with test outputs'

    call load_cifar10(training_images, training_images_dummy, &
      training_labels, training_labels_dummy, validation_images, &
      validation_labels, testing_images, testing_labels)

    res % ok = all(shape(testing_images) == [3072, 10000]) &
      .and. size(testing_labels) == 10000 &
      .and. valid_image_values(testing_images) &
      .and. valid_labels(testing_labels, 0., 9.)
  end function cifar10_loads_with_test_outputs

  type(test_result) function cifar100_loads_without_test_outputs() result(res)
    real, allocatable :: training_images(:,:), training_images_dummy(:,:)
    real, allocatable :: training_labels(:), training_labels_dummy(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)

    res % name = 'cifar100 loads without test outputs'

    call load_cifar100(training_images, training_images_dummy, &
      training_labels, training_labels_dummy, validation_images, &
      validation_labels)

    res % ok = all(shape(training_images) == [3072, 40000]) &
      .and. size(training_labels) == 40000 &
      .and. all(shape(validation_images) == [3072, 10000]) &
      .and. size(validation_labels) == 10000 &
      .and. valid_image_values(training_images) &
      .and. valid_image_values(validation_images) &
      .and. valid_labels(training_labels, 0., 99.) &
      .and. valid_labels(validation_labels, 0., 99.)
  end function cifar100_loads_without_test_outputs

  type(test_result) function cifar100_loads_with_test_outputs() result(res)
    real, allocatable :: training_images(:,:), training_images_dummy(:,:)
    real, allocatable :: training_labels(:), training_labels_dummy(:)
    real, allocatable :: validation_images(:,:), validation_labels(:)
    real, allocatable :: testing_images(:,:), testing_labels(:)

    res % name = 'cifar100 loads with test outputs'

    call load_cifar100(training_images, training_images_dummy, &
      training_labels, training_labels_dummy, validation_images, &
      validation_labels, testing_images, testing_labels)

    res % ok = all(shape(testing_images) == [3072, 10000]) &
      .and. size(testing_labels) == 10000 &
      .and. valid_image_values(testing_images) &
      .and. valid_labels(testing_labels, 0., 99.)
  end function cifar100_loads_with_test_outputs

  type(test_result) function cifar100_label_digits_are_one_hot() result(res)
    real :: encoded(100, 3)

    res % name = 'cifar100 label digits are one hot'
    encoded = label_digits_cifar100([0., 3., 99.])

    res % ok = all(shape(encoded) == [100, 3]) &
      .and. encoded(1, 1) == 1. &
      .and. encoded(4, 2) == 1. &
      .and. encoded(100, 3) == 1. &
      .and. sum(encoded) == 3.
  end function cifar100_label_digits_are_one_hot

  logical function valid_image_values(images)
    real, intent(in) :: images(:,:)

    valid_image_values = all(images >= 0.) .and. all(images <= 1.)
  end function valid_image_values

  logical function valid_labels(labels, min_label, max_label)
    real, intent(in) :: labels(:)
    real, intent(in) :: min_label, max_label

    valid_labels = all(labels >= min_label) .and. all(labels <= max_label) &
      .and. all(labels == real(int(labels)))
  end function valid_labels

end program test_datasets_cifar
