! Positive + CLI driver for qlight_parse_propagate_ids.
! --expect-fail semantics (IMPORTANT):
!   - parse REJECTS  -> Fortran error stop (nonzero) => negative test PASS
!   - parse ACCEPTS  -> stop 0 + PARSE_ACCEPTED     => negative test FAIL
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
      call qlight_parse_propagate_ids(trim(arg2), ids)
      ! Reached only if illegal input was wrongly accepted.
      write(*,*) 'PARSE_ACCEPTED'
      stop 0
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
