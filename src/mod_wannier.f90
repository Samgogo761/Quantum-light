module mod_wannier
  use mod_params
  implicit none

  integer :: nrpts = 0
  integer,     allocatable :: irvec(:,:)          ! (3, nrpts)
  integer,     allocatable :: ndegen(:)           ! (nrpts)
  complex(dp), allocatable :: Hmn_R(:,:,:)        ! (nwann, nwann, nrpts)
  complex(dp), allocatable :: rmn_R(:,:,:,:)      ! (nwann, nwann, 3, nrpts)
  real(dp),    allocatable :: Rvec_cart(:,:)       ! (3, nrpts) precomputed R in a.u.
  logical :: has_rmn = .false.

contains

  subroutine read_tb_file(filename)
    character(*), intent(in) :: filename
    integer :: u, ios, ir, m, n, nw_file, ios_parse
    integer :: r1, r2, r3, m_in, n_in
    real(dp) :: rr, ri, rx_r, rx_i, ry_r, ry_i, rz_r, rz_i
    character(256) :: line
    integer :: ir0
    real(dp) :: tb_a1_ang(3), tb_a2_ang(3), tb_a3_ang(3)

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot open ', trim(filename)
      error stop 1
    end if

    ! Line 1: comment/timestamp
    read(u, '(A)') line

    read(u, '(A)') line
    if (index(line, '.') > 0) then
      read(line, *, iostat=ios_parse) tb_a1_ang
      if (ios_parse /= 0) then
        write(*,*) 'ERROR: failed to parse lattice vector a1 in ', trim(filename)
        error stop 1
      end if
      read(u, *) tb_a2_ang
      read(u, *) tb_a3_ang
      call check_tb_lattice(filename, tb_a1_ang, tb_a2_ang, tb_a3_ang)
      read(u, *) nw_file
    else
      read(line, *) nw_file
    end if

    nwann = nw_file
    read(u, *) nrpts

    allocate(ndegen(nrpts))
    read(u, *) ndegen(1:nrpts)

    allocate(irvec(3, nrpts))
    allocate(Hmn_R(nwann, nwann, nrpts))
    allocate(rmn_R(nwann, nwann, 3, nrpts))
    Hmn_R = C_0
    rmn_R = C_0

    do ir = 1, nrpts
      read(u, '(A)') line
      read(u, *) r1, r2, r3
      irvec(:, ir) = [r1, r2, r3]
      do n = 1, nwann
        do m = 1, nwann
          read(u, *) m_in, n_in, rr, ri
          Hmn_R(m_in, n_in, ir) = cmplx(rr, ri, dp)
        end do
      end do
    end do

    do ir = 1, nrpts
      read(u, '(A)') line
      read(u, *) r1, r2, r3
      if (any([r1, r2, r3] /= irvec(:, ir))) then
        write(*,*) 'ERROR: inconsistent R header between H and r blocks in ', trim(filename)
        error stop 1
      end if
      do n = 1, nwann
        do m = 1, nwann
          read(u, *) m_in, n_in, rx_r, rx_i, ry_r, ry_i, rz_r, rz_i
          rmn_R(m_in, n_in, 1, ir) = cmplx(rx_r, rx_i, dp)
          rmn_R(m_in, n_in, 2, ir) = cmplx(ry_r, ry_i, dp)
          rmn_R(m_in, n_in, 3, ir) = cmplx(rz_r, rz_i, dp)
        end do
      end do
    end do
    close(u)

    Hmn_R = Hmn_R * eV_to_Ha
    rmn_R = rmn_R * Ang_to_bohr
    has_rmn = .true.

    ir0 = find_R0()
    if (ir0 > 0) then
      do n = 1, nwann
        Hmn_R(n, n, ir0) = Hmn_R(n, n, ir0) - cmplx(E_fermi, 0.0_dp, dp)
      end do
    end if

    call compute_Rvec_cart()
    write(*,'(A,I0,A,I0)') '  Wannier TB loaded: nwann=', nwann, ', nrpts=', nrpts
  end subroutine read_tb_file

  subroutine read_hr_file(filename)
    character(*), intent(in) :: filename
    integer :: u, ios, ir, m, n, nw_file, r1, r2, r3, m_in, n_in
    real(dp) :: rr, ri
    character(256) :: line
    integer :: idx, ir0

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot open ', trim(filename)
      error stop 1
    end if

    read(u, '(A)') line
    read(u, *) nw_file
    nwann = nw_file
    read(u, *) nrpts

    allocate(ndegen(nrpts))
    read(u, *) ndegen(1:nrpts)

    allocate(irvec(3, nrpts))
    allocate(Hmn_R(nwann, nwann, nrpts))
    Hmn_R = C_0

    idx = 0
    do ir = 1, nrpts
      do n = 1, nwann
        do m = 1, nwann
          read(u, *) r1, r2, r3, m_in, n_in, rr, ri
          if (m_in == 1 .and. n_in == 1) then
            idx = idx + 1
            irvec(1, idx) = r1
            irvec(2, idx) = r2
            irvec(3, idx) = r3
          end if
          Hmn_R(m_in, n_in, ir) = cmplx(rr, ri, dp)
        end do
      end do
    end do
    close(u)

    Hmn_R = Hmn_R * eV_to_Ha

    ir0 = find_R0()
    if (ir0 > 0) then
      do n = 1, nwann
        Hmn_R(n, n, ir0) = Hmn_R(n, n, ir0) - cmplx(E_fermi, 0.0_dp, dp)
      end do
    end if

    call compute_Rvec_cart()
    write(*,'(A,I0,A,I0)') '  Wannier HR loaded: nwann=', nwann, ', nrpts=', nrpts
  end subroutine read_hr_file

  subroutine read_r_file(filename)
    character(*), intent(in) :: filename
    integer :: u, ios, ir, m, n, nw_file, nrpts_r, m_in, n_in
    real(dp) :: rx_r, rx_i, ry_r, ry_i, rz_r, rz_i
    character(256) :: line
    integer :: r1, r2, r3
    integer, allocatable :: ndegen_r(:)

    open(newunit=u, file=filename, status='old', action='read', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'ERROR: cannot open ', trim(filename)
      error stop 1
    end if

    read(u, '(A)') line
    read(u, *) nw_file
    read(u, *) nrpts_r

    if (allocated(rmn_R)) then
      if (nrpts_r /= nrpts) then
        write(*,*) 'ERROR: nrpts mismatch between hr.dat and r.dat for ', trim(filename)
        error stop 1
      end if
    else
      nrpts = nrpts_r
    end if

    if (.not. allocated(rmn_R)) then
      allocate(rmn_R(nwann, nwann, 3, nrpts))
    end if
    rmn_R = C_0

    allocate(ndegen_r(nrpts_r))
    read(u, *) ndegen_r(1:nrpts_r)
    deallocate(ndegen_r)

    do ir = 1, nrpts_r
      do n = 1, nwann
        do m = 1, nwann
          read(u, *) r1, r2, r3, m_in, n_in, rx_r, rx_i, ry_r, ry_i, rz_r, rz_i
          rmn_R(m_in, n_in, 1, ir) = cmplx(rx_r, rx_i, dp)
          rmn_R(m_in, n_in, 2, ir) = cmplx(ry_r, ry_i, dp)
          rmn_R(m_in, n_in, 3, ir) = cmplx(rz_r, rz_i, dp)
        end do
      end do
    end do
    close(u)

    rmn_R = rmn_R * Ang_to_bohr
    has_rmn = .true.
    write(*,'(A,I0)') '  Wannier R loaded: nrpts=', nrpts_r
  end subroutine read_r_file

  subroutine compute_Rvec_cart()
    integer :: ir
    if (allocated(Rvec_cart)) deallocate(Rvec_cart)
    allocate(Rvec_cart(3, nrpts))
    do ir = 1, nrpts
      Rvec_cart(:, ir) = real(irvec(1,ir), dp) * a1 &
                        + real(irvec(2,ir), dp) * a2 &
                        + real(irvec(3,ir), dp) * a3
    end do
  end subroutine compute_Rvec_cart

  function find_R0() result(ir0)
    integer :: ir0, ir
    ir0 = 0
    do ir = 1, nrpts
      if (irvec(1,ir)==0 .and. irvec(2,ir)==0 .and. irvec(3,ir)==0) then
        ir0 = ir
        return
      end if
    end do
  end function find_R0

  subroutine check_tb_lattice(filename, tb_a1_ang, tb_a2_ang, tb_a3_ang)
    character(*), intent(in) :: filename
    real(dp), intent(in) :: tb_a1_ang(3), tb_a2_ang(3), tb_a3_ang(3)
    real(dp), parameter :: tol_ang = 1.0e-5_dp

    if (maxval(abs(tb_a1_ang - a1_ang)) > tol_ang .or. &
        maxval(abs(tb_a2_ang - a2_ang)) > tol_ang .or. &
        maxval(abs(tb_a3_ang - a3_ang)) > tol_ang) then
      write(*,'(A)') 'ERROR: lattice vectors in tb.dat do not match input.nml.'
      write(*,'(A)') '  File: '//trim(filename)
      write(*,'(A,3F18.10)') '  tb.dat a1 (Ang): ', tb_a1_ang
      write(*,'(A,3F18.10)') '  tb.dat a2 (Ang): ', tb_a2_ang
      write(*,'(A,3F18.10)') '  tb.dat a3 (Ang): ', tb_a3_ang
      write(*,'(A,3F18.10)') '  input  a1 (Ang): ', a1_ang
      write(*,'(A,3F18.10)') '  input  a2 (Ang): ', a2_ang
      write(*,'(A,3F18.10)') '  input  a3 (Ang): ', a3_ang
      error stop 1
    end if
  end subroutine check_tb_lattice

  subroutine fourier_ham(k_cart, Hk)
    real(dp),    intent(in)  :: k_cart(3)
    complex(dp), intent(out) :: Hk(nwann, nwann)
    integer :: ir
    real(dp) :: kdotR
    complex(dp) :: phase

    Hk = C_0
    do ir = 1, nrpts
      kdotR = dot_product(k_cart, Rvec_cart(:, ir))
      phase = exp(C_I * kdotR) / real(ndegen(ir), dp)
      Hk = Hk + phase * Hmn_R(:,:,ir)
    end do
  end subroutine fourier_ham

  subroutine fourier_dipole(k_cart, Dk)
    real(dp),    intent(in)  :: k_cart(3)
    complex(dp), intent(out) :: Dk(nwann, nwann, 3)
    integer :: ir, a
    real(dp) :: kdotR
    complex(dp) :: phase

    Dk = C_0
    do ir = 1, nrpts
      kdotR = dot_product(k_cart, Rvec_cart(:, ir))
      phase = exp(C_I * kdotR) / real(ndegen(ir), dp)
      do a = 1, 3
        Dk(:,:,a) = Dk(:,:,a) + phase * rmn_R(:,:,a,ir)
      end do
    end do
  end subroutine fourier_dipole

  subroutine fourier_velocity(k_cart, vk)
    real(dp),    intent(in)  :: k_cart(3)
    complex(dp), intent(out) :: vk(nwann, nwann, 3)
    integer :: ir, a
    real(dp) :: kdotR
    complex(dp) :: phase

    vk = C_0
    do ir = 1, nrpts
      kdotR = dot_product(k_cart, Rvec_cart(:, ir))
      phase = exp(C_I * kdotR) / real(ndegen(ir), dp)
      do a = 1, 3
        vk(:,:,a) = vk(:,:,a) + C_I * Rvec_cart(a, ir) * phase * Hmn_R(:,:,ir)
      end do
    end do
  end subroutine fourier_velocity

  subroutine fourier_all(k_cart, Hk, Dk, vk, phases)
    real(dp),    intent(in)  :: k_cart(3)
    complex(dp), intent(out) :: Hk(nwann, nwann)
    complex(dp), intent(out) :: Dk(nwann, nwann, 3)
    complex(dp), intent(out) :: vk(nwann, nwann, 3)
    complex(dp), intent(out) :: phases(nrpts)
    integer :: ir, a
    real(dp) :: kdotR

    Hk = C_0; Dk = C_0; vk = C_0
    do ir = 1, nrpts
      kdotR = dot_product(k_cart, Rvec_cart(:, ir))
      phases(ir) = exp(C_I * kdotR) / real(ndegen(ir), dp)
      Hk = Hk + phases(ir) * Hmn_R(:,:,ir)
      do a = 1, 3
        Dk(:,:,a) = Dk(:,:,a) + phases(ir) * rmn_R(:,:,a,ir)
        vk(:,:,a) = vk(:,:,a) + C_I * Rvec_cart(a,ir) * phases(ir) * Hmn_R(:,:,ir)
      end do
    end do
  end subroutine fourier_all

  subroutine fourier_ham_with_phases(phases_in, Hk)
    complex(dp), intent(in)  :: phases_in(nrpts)
    complex(dp), intent(out) :: Hk(nwann, nwann)
    integer :: ir
    Hk = C_0
    do ir = 1, nrpts
      Hk = Hk + phases_in(ir) * Hmn_R(:,:,ir)
    end do
  end subroutine fourier_ham_with_phases

end module mod_wannier
