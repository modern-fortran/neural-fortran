module tuff
  ! Testing Unframework for Fortran (TUFF)
  use iso_fortran_env, only: stderr => error_unit, stdout => output_unit
  implicit none

  private
  public :: test, test_result

  type :: test_result
    character(:), allocatable :: name
    logical :: ok = .true.
    real :: elapsed = 0.
  end type test_result

  interface test
    module procedure test_logical
    module procedure test_func
    module procedure test_array
  end interface test

  abstract interface
    function func() result(res)
      import :: test_result
      type(test_result) :: res
    end function func
  end interface

contains

  type(test_result) function test_logical(name, cond) result(res)
    ! Test a single logical expression.
    character(*), intent(in) :: name
    logical, intent(in) :: cond
    res % name = name
    res % ok = .true.
    res % elapsed = 0.
    if (.not. cond) then
      write(stderr, '(a)') 'Test ' // trim(name) // ' failed.'
      res % ok = .false.
    end if
  end function test_logical


  type(test_result) function test_func(f) result(res)
    ! Test a user-provided function f that returns a test_result.
    ! f is responsible for setting the test name and the ok field. 
    procedure(func) :: f
    real :: t1, t2
    res % name = ''
    call cpu_time(t1)
    res = f()
    call cpu_time(t2)
    res % elapsed = t2 - t1
    if (len_trim(res % name) == 0) res % name = 'Anonymous test'
    if (.not. res % ok) then
      write(stderr, '(a, f6.3)') 'Test failed: ' //  trim(res % name)
    end if
  end function test_func


  type(test_result) function test_array(name, tests) result(suite)
    ! Test a suite of tests, each of which is a test_result.
    character(*), intent(in) :: name
    type(test_result), intent(in) :: tests(:)
    suite % ok = all(tests % ok)
    suite % elapsed = sum(tests % elapsed)
    if (.not. suite % ok) then
      ! Report to stderr only on failure.
      write(stderr, '(i0,a,i0,a)') count(.not. tests % ok), '/', size(tests), &
        " tests failed in suite: " // trim(name)
    end if
  end function test_array

end module tuff