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

  subroutine write_current_decomposed(filename, Jt_tot, Jt_intra_in, Jt_inter_in, nt_in, dt_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: Jt_tot(:,:), Jt_intra_in(:,:), Jt_inter_in(:,:)
    integer,      intent(in) :: nt_in
    real(dp),     intent(in) :: dt_in
    integer :: u, it

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# it  time(fs)  Jx_intra  Jy_intra  Jx_inter  Jy_inter  Jx_tot  Jy_tot'
    do it = 1, nt_in
      write(u, '(I8, 7ES18.10)') it, (it-1)*dt_in*au_to_fs, &
        Jt_intra_in(it,1), Jt_intra_in(it,2), &
        Jt_inter_in(it,1), Jt_inter_in(it,2), &
        Jt_tot(it,1), Jt_tot(it,2)
    end do
    close(u)
  end subroutine write_current_decomposed

  subroutine write_valley_current(filename, Jt_K_in, Jt_Kp_in, nt_in, dt_in)
    character(*), intent(in) :: filename
    real(dp),     intent(in) :: Jt_K_in(:,:), Jt_Kp_in(:,:)
    integer,      intent(in) :: nt_in
    real(dp),     intent(in) :: dt_in
    integer :: u, it
    real(dp) :: Jx_tot, Jy_tot, eta_x, eta_y

    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# it  time(fs)  Jx_K  Jy_K  Jx_Kp  Jy_Kp  eta_x  eta_y'
    do it = 1, nt_in
      Jx_tot = Jt_K_in(it,1) + Jt_Kp_in(it,1)
      Jy_tot = Jt_K_in(it,2) + Jt_Kp_in(it,2)
      if (abs(Jx_tot) > 1.0e-30_dp) then
        eta_x = (Jt_K_in(it,1) - Jt_Kp_in(it,1)) / Jx_tot
      else
        eta_x = 0.0_dp
      end if
      if (abs(Jy_tot) > 1.0e-30_dp) then
        eta_y = (Jt_K_in(it,2) - Jt_Kp_in(it,2)) / Jy_tot
      else
        eta_y = 0.0_dp
      end if
      write(u, '(I8, 7ES18.10)') it, (it-1)*dt_in*au_to_fs, &
        Jt_K_in(it,1), Jt_K_in(it,2), &
        Jt_Kp_in(it,1), Jt_Kp_in(it,2), eta_x, eta_y
    end do
    close(u)
  end subroutine write_valley_current

end module mod_current
