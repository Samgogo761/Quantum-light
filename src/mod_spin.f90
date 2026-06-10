module mod_spin
  !---------------------------------------------------------------------------
  ! Spin-resolved (S_z) current for the length-gauge path.
  !
  ! Spin current operator (z-spin, Cartesian current direction a):
  !     J^{s_z}_a(k) = (1/2) { S_z(k), v_a(k) }      (symmetrised)
  ! so  J^{s_z}_a(t) = -1/(Nk A) * Re sum_k Tr[ JS_a(k) rho(k,t) ],
  ! with JS_a = 1/2 ( S_z v_a + v_a S_z ) precomputed in the band basis.
  !
  ! S_z source (PLUGGABLE):
  !   * default  : nominal diagonal S_z = diag(+1/2,-1/2,...) in the spinor
  !                Wannier basis (spin is the fast index => 'interleaved').
  !                Exact only if the MLWFs are spin eigenstates; with SOC this
  !                is approximate (good for Cr-d, rough for I-p).
  !   * spin_sz_file : read a real S_z(Wannier) matrix (e.g. derived from a
  !                wannier90 .spn run) -> rigorous. Format: lines "m n Re Im".
  !
  ! report_equilibrium_spin() prints <n,k|S_z|n,k> band averages as a sanity
  ! check: for the AFM the two Cr sublattices should carry opposite sign and the
  ! occupied manifold should net ~0.
  !---------------------------------------------------------------------------
  use mod_params
  use mod_wannier, only: nwann
  use mod_crystal, only: U_trunc, Ek, project_to_trunc
  implicit none

  complex(dp), allocatable :: SzVk_eq(:,:,:,:,:)   ! (n_trunc,n_trunc,2,nkx,nky) spin-velocity
  real(dp),    allocatable :: Sz_band_avg(:)       ! (n_trunc) BZ-averaged <S_z> per band
  real(dp),    allocatable :: Jt_spin(:,:)         ! (nt,2) spin-z current
  logical :: spin_ready = .false.

contains

  subroutine build_sz_wannier(Sz_wann)
    complex(dp), intent(out) :: Sz_wann(nwann, nwann)
    integer :: i, m, n, u, ios
    real(dp) :: re, im
    character(256) :: line

    Sz_wann = C_0

    if (len_trim(spin_sz_file) > 0) then
      open(newunit=u, file=trim(spin_sz_file), status='old', action='read', iostat=ios)
      if (ios /= 0) then
        write(*,*) 'ERROR: cannot open spin_sz_file: ', trim(spin_sz_file)
        error stop 1
      end if
      do
        read(u, '(A)', iostat=ios) line
        if (ios /= 0) exit
        if (len_trim(line) == 0) cycle
        if (line(1:1) == '#') cycle
        read(line, *, iostat=ios) m, n, re, im
        if (ios /= 0) cycle
        if (m >= 1 .and. m <= nwann .and. n >= 1 .and. n <= nwann) &
          Sz_wann(m, n) = cmplx(re, im, dp)
      end do
      close(u)
      write(*,'(A,A)') '  S_z(Wannier) read from ', trim(spin_sz_file)
    else
      ! Nominal diagonal S_z = +-1/2.
      if (trim(spin_order) == 'blocked') then
        do i = 1, nwann
          if (i <= nwann/2) then
            Sz_wann(i, i) = cmplx(0.5_dp, 0.0_dp, dp)
          else
            Sz_wann(i, i) = cmplx(-0.5_dp, 0.0_dp, dp)
          end if
        end do
      else   ! 'interleaved': (up,down,up,down,...)
        do i = 1, nwann
          if (mod(i, 2) == 1) then
            Sz_wann(i, i) = cmplx(0.5_dp, 0.0_dp, dp)
          else
            Sz_wann(i, i) = cmplx(-0.5_dp, 0.0_dp, dp)
          end if
        end do
      end if
      write(*,'(A,A,A)') '  S_z(Wannier) = nominal diag(+-1/2), order=', &
        trim(spin_order), ' (approximate under SOC; use spin_sz_file for rigour)'
    end if
  end subroutine build_sz_wannier

  subroutine setup_spin_operator(Pk_eq_in)
    ! Pk_eq_in: band-basis covariant velocity (n_trunc,n_trunc,3,nkx,nky) from mod_sbe.
    complex(dp), intent(in) :: Pk_eq_in(n_trunc, n_trunc, 3, nkx, nky)
    complex(dp), allocatable :: Sz_wann(:,:), Sz_k(:,:), Pa(:,:), SzP(:,:), PSz(:,:)
    integer :: ikx, iky, a, n
    real(dp), allocatable :: cnt(:)

    allocate(Sz_wann(nwann, nwann))
    call build_sz_wannier(Sz_wann)

    if (allocated(SzVk_eq)) deallocate(SzVk_eq)
    allocate(SzVk_eq(n_trunc, n_trunc, 2, nkx, nky))
    allocate(Sz_band_avg(n_trunc)); Sz_band_avg = 0.0_dp
    allocate(cnt(n_trunc));         cnt = 0.0_dp
    SzVk_eq = C_0

    !$OMP PARALLEL DEFAULT(shared) PRIVATE(ikx, iky, a, n, Sz_k, Pa, SzP, PSz) &
    !$OMP   REDUCTION(+:Sz_band_avg)
    allocate(Sz_k(n_trunc, n_trunc), Pa(n_trunc, n_trunc))
    allocate(SzP(n_trunc, n_trunc), PSz(n_trunc, n_trunc))
    !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
    do iky = 1, nky
      do ikx = 1, nkx
        ! S_z in band basis at this k
        call project_to_trunc(Sz_wann, ikx, iky, Sz_k)
        do n = 1, n_trunc
          Sz_band_avg(n) = Sz_band_avg(n) + real(Sz_k(n, n), dp)
        end do
        ! spin-velocity JS_a = 1/2 (Sz Pa + Pa Sz)
        do a = 1, 2
          Pa = Pk_eq_in(:, :, a, ikx, iky)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Sz_k,n_trunc,Pa,n_trunc,C_0,SzP,n_trunc)
          call zgemm('N','N',n_trunc,n_trunc,n_trunc,C_1,Pa,n_trunc,Sz_k,n_trunc,C_0,PSz,n_trunc)
          SzVk_eq(:, :, a, ikx, iky) = 0.5_dp * (SzP + PSz)
        end do
      end do
    end do
    !$OMP END DO
    deallocate(Sz_k, Pa, SzP, PSz)
    !$OMP END PARALLEL

    Sz_band_avg = Sz_band_avg / real(nkx * nky, dp)
    deallocate(Sz_wann, cnt)
    spin_ready = .true.
    call report_equilibrium_spin()
  end subroutine setup_spin_operator

  subroutine report_equilibrium_spin()
    integer :: n, nv_lo, nv_hi
    real(dp) :: net_val, net_all, max_abs

    net_val = 0.0_dp; net_all = 0.0_dp; max_abs = 0.0_dp
    nv_lo = 1; nv_hi = nv
    do n = 1, n_trunc
      if (n >= nv_lo .and. n <= nv_hi) net_val = net_val + Sz_band_avg(n)
      net_all = net_all + Sz_band_avg(n)
      max_abs = max(max_abs, abs(Sz_band_avg(n)))
    end do
    write(*,'(A)')        '  Equilibrium spin check (BZ-averaged <S_z> per band):'
    write(*,'(A,F10.5)')  '    max |<S_z>_n|                 : ', max_abs
    write(*,'(A,F10.5)')  '    sum over valence bands (1..nv): ', net_val
    write(*,'(A,F10.5)')  '    sum over all kept bands       : ', net_all
    write(*,'(A)')        '    (AFM: valence sum should be ~0; bands strongly +/- polarised)'
  end subroutine report_equilibrium_spin

  subroutine write_spin_current(filename, nt_in, dt_in)
    character(*), intent(in) :: filename
    integer,      intent(in) :: nt_in
    real(dp),     intent(in) :: dt_in
    integer :: u, it
    if (.not. allocated(Jt_spin)) return
    open(newunit=u, file=filename, status='replace', action='write')
    write(u, '(A)') '# spin-z current J^{s_z}(t) = (1/2)<{S_z,v}>  (a.u.)'
    write(u, '(A)') '# it  time(fs)  Jx_spin  Jy_spin'
    do it = 1, nt_in
      write(u, '(I7, 3ES18.8)') it, real(it-1,dp)*dt_in*au_to_fs, &
        Jt_spin(it, 1), Jt_spin(it, 2)
    end do
    close(u)
    write(*,'(A,A)') '  Spin-z current written to ', trim(filename)
  end subroutine write_spin_current

end module mod_spin
