module nf_metric

  !! This module provides a collection of metric functions.

  implicit none

  private
  public :: metric_type

  type, abstract :: metric_type
  contains
    procedure(metric_interface), nopass, deferred :: eval
  end type metric_type

  abstract interface
    pure function metric_interface(true, predicted) result(res)
      real, intent(in) :: true(:)
      real, intent(in) :: predicted(:)
      real :: res
    end function metric_interface
  end interface

  type, extends(metric_type) :: corr
    !! Pearson correlation
  contains
    procedure, nopass :: eval => corr_eval
  end type corr

  contains

  pure module function corr_eval(true, predicted) result(res)
    !! Pearson correlation function:
    !!
    real, intent(in) :: true(:)
      !! True values, i.e. labels from training datasets
    real, intent(in) :: predicted(:)
      !! Values predicted by the network
    real :: res
      !! Resulting loss value


    real :: m_true, m_pred

    m_true = sum(true) / size(true)
    m_pred = sum(predicted) / size(predicted)

    res = dot_product(true - m_true, predicted - m_pred) / &
      sqrt(sum((true - m_true)**2)*sum((predicted - m_pred)**2))

  end function corr_eval

end module nf_metric
