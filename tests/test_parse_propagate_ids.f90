! Positive + self-contained negative-driver tests for qlight_parse_propagate_ids.
! Negatives are exercised by spawning this binary with --expect-fail <list>.
program test_parse_propagate_ids
  use mod_quantum_light, only: qlight_parse_propagate_ids, dp
  implicit none

  character(256) :: arg1, arg2
  integer :: narg
  integer, allocatable :: ids(:)
  logical :: pass_pos

  narg = command_argument_count()
  if (narg >= 1) then
    call get_command_argument(1, arg1)
    if (trim(arg1) == '--expect-fail') then
      if (narg < 2) then
        write(*,*) 'ERROR: --expect-fail needs a list argument'
        stop 2
      end if
      call get_command_argument(2, arg2)
      ! Must error-stop / non-zero for invalid lists.
      call qlight_parse_propagate_ids(trim(arg2), ids)
      write(*,*) 'ERROR: expected failure for list: ', trim(arg2)
      stop 1
    else if (trim(arg1) == '--expect-ok') then
      if (narg < 2) stop 2
      call get_command_argument(2, arg2)
      call qlight_parse_propagate_ids(trim(arg2), ids)
      write(*,'(A,I0)') 'n_ids=', size(ids)
      stop 0
    end if
  end if

  pass_pos = .true.
  call qlight_parse_propagate_ids('1,2,3', ids)
  if (size(ids) /= 3) pass_pos = .false.
  if (ids(1) /= 1 .or. ids(2) /= 2 .or. ids(3) /= 3) pass_pos = .false.
  deallocate(ids)

  call qlight_parse_propagate_ids(' 5, 7 ,9 ', ids)
  if (size(ids) /= 3) pass_pos = .false.
  if (ids(1) /= 5 .or. ids(2) /= 7 .or. ids(3) /= 9) pass_pos = .false.
  deallocate(ids)

  call qlight_parse_propagate_ids('42', ids)
  if (size(ids) /= 1 .or. ids(1) /= 42) pass_pos = .false.
  deallocate(ids)

  if (pass_pos) then
    print '(A)', 'RESULT[parse_ids_pos]: PASS'
    stop 0
  else
    print '(A)', 'RESULT[parse_ids_pos]: FAIL'
    stop 1
  end if
end program test_parse_propagate_ids
