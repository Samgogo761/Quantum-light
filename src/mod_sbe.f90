module mod_sbe
  use mod_params
  use mod_wannier
  use mod_crystal
  use mod_laser
  implicit none

  complex(dp), allocatable :: rho(:,:,:,:)      ! (n_trunc, n_trunc, nkx, nky)
  complex(dp), allocatable :: rho_eq(:,:,:,:)   ! equilibrium density matrix
  real(dp),    allocatable :: Jt(:,:)           ! (nt, 2) current output
  real(dp) :: deph_factor

contains

  subroutine init_density_matrix()
    integer :: ikx, iky, n

    if (allocated(rho)) deallocate(rho, rho_eq)
    allocate(rho(n_trunc, n_trunc, nkx, nky))
    allocate(rho_eq(n_trunc, n_trunc, nkx, nky))
    rho = C_0
    do iky = 1, nky
      do ikx = 1, nkx
        do n = 1, nv
          rho(n, n, ikx, iky) = C_1
        end do
      end do
    end do
    rho_eq = rho

    deph_factor = exp(-dt * real(n_dt_deph, dp) / T2)

    if (allocated(Jt)) deallocate(Jt)
    allocate(Jt(nt, 2))
    Jt = 0.0_dp
  end subroutine init_density_matrix

  subroutine propagate()
    integer  :: it, ikx, iky, a, m, n
    real(dp) :: Jx_it, Jy_it, Nk_inv, k_t(3)

    complex(dp), allocatable :: Hk_W(:,:), Dk_W(:,:,:), vk_W(:,:,:)
    complex(dp), allocatable :: Hk_proj(:,:), Dk_proj(:,:,:), vk_proj(:,:,:)
    complex(dp), allocatable :: Ht(:,:), rho_k(:,:), rho_new(:,:)
    complex(dp), allocatable :: phases(:)
    complex(dp), allocatable :: phase0(:,:,:)
    complex(dp), allocatable :: phase_A(:)
    complex(dp) :: tr_val

    Nk_inv = 1.0_dp / real(nkx * nky, dp)

    allocate(phase0(nrpts, nkx, nky))
    call precompute_phase0(phase0)

    do it = 1, nt
      Jx_it = 0.0_dp
      Jy_it = 0.0_dp

      !$OMP PARALLEL DEFAULT(shared) &
      !$OMP PRIVATE(ikx, iky, a, m, n, k_t, tr_val, &
      !$OMP         Hk_W, Dk_W, vk_W, Hk_proj, Dk_proj, vk_proj, &
      !$OMP         Ht, rho_k, rho_new, phases, phase_A) &
      !$OMP REDUCTION(+:Jx_it, Jy_it)

      allocate(Hk_W(nwann, nwann), Dk_W(nwann, nwann, 3), vk_W(nwann, nwann, 3))
      allocate(Hk_proj(n_trunc, n_trunc), Dk_proj(n_trunc, n_trunc, 3))
      allocate(vk_proj(n_trunc, n_trunc, 3))
      allocate(Ht(n_trunc, n_trunc), rho_k(n_trunc, n_trunc), rho_new(n_trunc, n_trunc))
      allocate(phases(nrpts), phase_A(nrpts))

      call compute_phase_A(At_vec(it,:), phase_A)

      !$OMP DO COLLAPSE(2) SCHEDULE(dynamic)
      do iky = 1, nky
        do ikx = 1, nkx

          phases(:) = phase0(:, ikx, iky) * phase_A(:)

          call fourier_all_with_phases(phases, Hk_W, Dk_W, vk_W)

          call project_to_trunc(Hk_W, ikx, iky, Hk_proj)

          Ht = Hk_proj
          if (has_rmn) then
            call project_vec_to_trunc(Dk_W, ikx, iky, Dk_proj)
            do a = 1, 3
              if (abs(Et_vec(it, a)) > 1.0e-30_dp) then
                Ht = Ht - Et_vec(it, a) * Dk_proj(:,:,a)
              end if
            end do
          end if

          rho_k = rho(:,:,ikx,iky)
          call rk4_step(Ht, rho_k, dt, rho_new)

          if (mod(it, n_dt_deph) == 0) then
            call apply_dephasing(rho_new, rho_eq(:,:,ikx,iky))
          end if

          rho(:,:,ikx,iky) = rho_new

          call project_vec_to_trunc(vk_W, ikx, iky, vk_proj)

          do a = 1, 2
            tr_val = C_0
            do n = 1, n_trunc
              do m = 1, n_trunc
                tr_val = tr_val + vk_proj(m, n, a) * rho_new(n, m)
              end do
            end do
            if (a == 1) Jx_it = Jx_it + real(tr_val, dp)
            if (a == 2) Jy_it = Jy_it + real(tr_val, dp)
          end do

        end do
      end do
      !$OMP END DO

      deallocate(Hk_W, Dk_W, vk_W, Hk_proj, Dk_proj, vk_proj)
      deallocate(Ht, rho_k, rho_new, phases, phase_A)
      !$OMP END PARALLEL

      Jt(it, 1) = -Jx_it * Nk_inv
      Jt(it, 2) = -Jy_it * Nk_inv

      if (mod(it, max(nt/20, 1)) == 0) then
        write(*,'(A,I0,A,I0,A,F6.1,A)') '  Step ', it, ' / ', nt, &
          ' (', 100.0_dp*real(it,dp)/real(nt,dp), '%)'
      end if
    end do

    deallocate(phase0)
  end subroutine propagate

  subroutine precompute_phase0(phase0)
    complex(dp), intent(out) :: phase0(nrpts, nkx, nky)
    integer :: ikx, iky, ir
    real(dp) :: kdotR

    !$OMP PARALLEL DO COLLAPSE(2) PRIVATE(ir, kdotR)
    do iky = 1, nky
      do ikx = 1, nkx
        do ir = 1, nrpts
          kdotR = dot_product(kpts_cart(:, ikx, iky), Rvec_cart(:, ir))
          phase0(ir, ikx, iky) = exp(C_I * kdotR) / real(ndegen(ir), dp)
        end do
      end do
    end do
    !$OMP END PARALLEL DO
  end subroutine precompute_phase0

  subroutine compute_phase_A(At_now, phase_A)
    real(dp),    intent(in)  :: At_now(3)
    complex(dp), intent(out) :: phase_A(nrpts)
    integer :: ir
    do ir = 1, nrpts
      phase_A(ir) = exp(C_I * dot_product(At_now, Rvec_cart(:, ir)))
    end do
  end subroutine compute_phase_A

  subroutine fourier_all_with_phases(phases_in, Hk, Dk, vk)
    complex(dp), intent(in)  :: phases_in(nrpts)
    complex(dp), intent(out) :: Hk(nwann, nwann)
    complex(dp), intent(out) :: Dk(nwann, nwann, 3)
    complex(dp), intent(out) :: vk(nwann, nwann, 3)
    integer :: ir, a

    Hk = C_0; Dk = C_0; vk = C_0
    do ir = 1, nrpts
      Hk = Hk + phases_in(ir) * Hmn_R(:,:,ir)
      do a = 1, 3
        if (has_rmn) then
          Dk(:,:,a) = Dk(:,:,a) + phases_in(ir) * rmn_R(:,:,a,ir)
        end if
        vk(:,:,a) = vk(:,:,a) + C_I * Rvec_cart(a,ir) * phases_in(ir) * Hmn_R(:,:,ir)
      end do
    end do
  end subroutine fourier_all_with_phases

  subroutine rk4_step(Ht, rho_in, dt_step, rho_out)
    complex(dp), intent(in)  :: Ht(:,:), rho_in(:,:)
    real(dp),    intent(in)  :: dt_step
    complex(dp), intent(out) :: rho_out(:,:)
    integer :: ns
    complex(dp), allocatable :: k1(:,:), k2(:,:), k3(:,:), k4(:,:), rho_tmp(:,:)

    ns = size(rho_in, 1)
    allocate(k1(ns,ns), k2(ns,ns), k3(ns,ns), k4(ns,ns), rho_tmp(ns,ns))

    call commutator(Ht, rho_in, k1, ns)
    rho_tmp = rho_in + 0.5_dp * dt_step * k1
    call commutator(Ht, rho_tmp, k2, ns)
    rho_tmp = rho_in + 0.5_dp * dt_step * k2
    call commutator(Ht, rho_tmp, k3, ns)
    rho_tmp = rho_in + dt_step * k3
    call commutator(Ht, rho_tmp, k4, ns)

    rho_out = rho_in + (dt_step / 6.0_dp) * (k1 + 2.0_dp*k2 + 2.0_dp*k3 + k4)

    deallocate(k1, k2, k3, k4, rho_tmp)
  end subroutine rk4_step

  subroutine commutator(A, B, comm, ns)
    integer,     intent(in)  :: ns
    complex(dp), intent(in)  :: A(ns, ns), B(ns, ns)
    complex(dp), intent(out) :: comm(ns, ns)
    complex(dp) :: AB(ns, ns)

    call zgemm('N', 'N', ns, ns, ns, C_1, A, ns, B, ns, C_0, AB, ns)
    call zgemm('N', 'N', ns, ns, ns, C_1, B, ns, A, ns, C_0, comm, ns)
    comm = -C_I * (AB - comm)
  end subroutine commutator

  subroutine apply_dephasing(rho_k, rho_eq_k)
    complex(dp), intent(inout) :: rho_k(:,:)
    complex(dp), intent(in)    :: rho_eq_k(:,:)
    integer :: m, n, ns
    ns = size(rho_k, 1)
    do n = 1, ns
      do m = 1, ns
        if (m /= n) rho_k(m,n) = rho_k(m,n) * deph_factor
      end do
    end do
  end subroutine apply_dephasing

  subroutine run_single_trajectory(Jt_out)
    real(dp), intent(out) :: Jt_out(:,:)
    call init_density_matrix()
    call propagate()
    Jt_out = Jt
  end subroutine run_single_trajectory

end module mod_sbe
