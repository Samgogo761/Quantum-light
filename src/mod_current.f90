module mod_current
  use mod_params
  implicit none

contains

  subroutine write_current(filename, Jt, nt_in, dt_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: Jt(:,:)
    integer,      intent(in) :: nt_in
    real(dp),     intent(in) :: dt_in
    integer :: u, it

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# it   time(fs)   Jx(a.u.)   Jy(a.u.)'
    do it = 1, nt_in
      write(u, '(I8, 3ES18.10)') it, (it-1)*dt_in*au_to_fs, Jt(it, 1), Jt(it, 2)
    end do
    close(u)
  end subroutine write_current

end module mod_current
