#!/usr/bin/env python3
"""Patch a QE 7.2 pw2wannier90.f90 copy to read UPF-defined radials.

Does not touch the production QE tree. The patched binary requires
seedname.radial_id and seedname.radial_R; it will not silently fall back
to hydrogenic zona.
"""
from __future__ import annotations

import argparse
from pathlib import Path

CALL_OLD = "      CALL radialpart(npw, qg, alpha_w(iw), r_w(iw), lmax, radial)"
CALL_NEW = "      CALL radialpart(npw, qg, alpha_w(iw), r_w(iw), lmax, radial, iw)"

SUB_OLD = """SUBROUTINE radialpart(ng, q, alfa, rvalue, lmax, radial)
  !-----------------------------------------------------------------------
  !
  ! This routine computes a table with the radial Fourier transform
  ! of the radial functions.
  !
  USE kinds,      ONLY : dp
  USE constants,  ONLY : fpi
  USE cell_base,  ONLY : omega
  !
  IMPLICIT NONE
  ! I/O
  INTEGER :: ng, rvalue, lmax
  real(DP) :: q(ng), alfa, radial(ng,0:lmax)"""

SUB_NEW = """SUBROUTINE radialpart(ng, q, alfa, rvalue, lmax, radial, iw)
  !-----------------------------------------------------------------------
  !
  ! Radial Fourier transform of the radial functions.
  ! UPF-defined R(r) tables are required (scheme C). Missing tables abort.
  !
  USE kinds,      ONLY : dp
  USE constants,  ONLY : fpi
  USE cell_base,  ONLY : omega
  USE io_global,  ONLY : stdout, ionode
  USE wannier,    ONLY : seedname, n_proj
  !
  IMPLICIT NONE
  ! I/O
  INTEGER :: ng, rvalue, lmax, iw
  real(DP) :: q(ng), alfa, radial(ng,0:lmax)"""

DECL_OLD = """  INTEGER :: l, lp1, ir, ig, mesh_r
  real(DP), ALLOCATABLE :: bes(:), func_r(:), r(:), rij(:), aux(:)
"""
DECL_NEW = """  INTEGER :: l, lp1, ir, ig, mesh_r
  real(DP), ALLOCATABLE :: bes(:), func_r(:), r(:), rij(:), aux(:)
  LOGICAL, SAVE :: custom_active = .false.
  INTEGER, SAVE :: n_id = 0, n_type = 0, mesh_c = 0
  INTEGER, ALLOCATABLE, SAVE :: id_of_iw(:)
  real(DP), ALLOCATABLE, SAVE :: r_upf(:), func_upf(:,:)
"""

BODY_OLD = """  ALLOCATE ( bes(mesh_r), func_r(mesh_r), r(mesh_r), rij(mesh_r) )
  ALLOCATE ( aux(mesh_r))
  !
  !    compute the radial mesh
  !
  DO ir = 1, mesh_r
     x = xmin  + dble (ir - 1) * dx
     r (ir) = exp (x) / alfa
     rij (ir) = dx  * r (ir)
  ENDDO
  !
  IF (rvalue==1) func_r(:) = 2.d0 * alfa**(3.d0/2.d0) * exp(-alfa*r(:))
  IF (rvalue==2) func_r(:) = 1.d0/sqrt(8.d0) * alfa**(3.d0/2.d0) * &
                     (2.0d0 - alfa*r(:)) * exp(-alfa*r(:)*0.5d0)
  IF (rvalue==3) func_r(:) = sqrt(4.d0/27.d0) * alfa**(3.0d0/2.0d0) * &
                     (1.d0 - 2.0d0/3.0d0*alfa*r(:) + 2.d0*(alfa*r(:))**2/27.d0) * &
                                           exp(-alfa*r(:)/3.0d0)
  pref = fpi/sqrt(omega)
"""

BODY_NEW = """  ALLOCATE ( bes(mesh_r), func_r(mesh_r), r(mesh_r), rij(mesh_r) )
  ALLOCATE ( aux(mesh_r))
  CALL load_upf_custom_radials()
  IF (custom_active) THEN
     IF (iw < 1 .or. iw > n_id) CALL errore('radialpart','iw out of radial_id', iw)
     IF (id_of_iw(iw) < 1 .or. id_of_iw(iw) > n_type) &
        CALL errore('radialpart','radial type out of range', id_of_iw(iw))
     IF (mesh_c /= mesh_r) CALL errore('radialpart','UPF radial mesh mismatch', mesh_c)
     r(1:mesh_r) = r_upf(1:mesh_r)
     rij(1:mesh_r) = dx * r(1:mesh_r)
     func_r(1:mesh_r) = func_upf(1:mesh_r, id_of_iw(iw))
  ELSE
     DO ir = 1, mesh_r
        x = xmin  + dble (ir - 1) * dx
        r (ir) = exp (x) / alfa
        rij (ir) = dx  * r (ir)
     ENDDO
     IF (rvalue==1) func_r(:) = 2.d0 * alfa**(3.d0/2.d0) * exp(-alfa*r(:))
     IF (rvalue==2) func_r(:) = 1.d0/sqrt(8.d0) * alfa**(3.d0/2.d0) * &
                        (2.0d0 - alfa*r(:)) * exp(-alfa*r(:)*0.5d0)
     IF (rvalue==3) func_r(:) = sqrt(4.d0/27.d0) * alfa**(3.0d0/2.0d0) * &
                        (1.d0 - 2.0d0/3.0d0*alfa*r(:) + 2.d0*(alfa*r(:))**2/27.d0) * &
                                              exp(-alfa*r(:)/3.0d0)
  ENDIF
  pref = fpi/sqrt(omega)
"""

LOADER = """
CONTAINS
  SUBROUTINE load_upf_custom_radials()
    IMPLICIT NONE
    INTEGER :: iu, ir, it, ios, l_dummy
    LOGICAL :: have_id, have_R
    CHARACTER(len=16) :: tag
    CHARACTER(len=256) :: filid, filr
    LOGICAL, SAVE :: custom_loaded = .false.
    IF (custom_loaded) RETURN
    custom_loaded = .true.
    custom_active = .false.
    filid = trim(seedname)//'.radial_id'
    filr  = trim(seedname)//'.radial_R'
    INQUIRE(file=filid, exist=have_id)
    INQUIRE(file=filr,  exist=have_R)
    IF (.not. have_id) CALL errore('radialpart', 'missing '//trim(filid), 1)
    IF (.not. have_R)  CALL errore('radialpart', 'missing '//trim(filr), 1)
    iu = 417
    OPEN(unit=iu, file=filid, status='old', form='formatted', action='read', iostat=ios)
    IF (ios /= 0) CALL errore('radialpart', 'open radial_id', ios)
    READ(iu, *, iostat=ios) n_id
    IF (ios /= 0) CALL errore('radialpart', 'read n_id', ios)
    IF (n_id /= n_proj) CALL errore('radialpart', 'radial_id count != n_proj', n_id)
    ALLOCATE(id_of_iw(n_id))
    READ(iu, *, iostat=ios) (id_of_iw(ir), ir=1, n_id)
    IF (ios /= 0) CALL errore('radialpart', 'read radial_id body', ios)
    CLOSE(iu)
    OPEN(unit=iu, file=filr, status='old', form='formatted', action='read', iostat=ios)
    IF (ios /= 0) CALL errore('radialpart', 'open radial_R', ios)
    READ(iu, *, iostat=ios) n_type, mesh_c
    IF (ios /= 0) CALL errore('radialpart', 'read radial_R header', ios)
    ALLOCATE(r_upf(mesh_c), func_upf(mesh_c, n_type))
    READ(iu, *, iostat=ios) (r_upf(ir), ir=1, mesh_c)
    IF (ios /= 0) CALL errore('radialpart', 'read r mesh', ios)
    DO it = 1, n_type
       READ(iu, *, iostat=ios) tag, l_dummy
       IF (ios /= 0) CALL errore('radialpart', 'read radial tag', it)
       READ(iu, *, iostat=ios) (func_upf(ir, it), ir=1, mesh_c)
       IF (ios /= 0) CALL errore('radialpart', 'read R(r)', it)
    ENDDO
    CLOSE(iu)
    custom_active = .true.
    IF (ionode) THEN
       WRITE(stdout,'(a,i4,a,i5,a,i4)') &
          ' UPF_CUSTOM_RADIAL_LOADED n_type=', n_type, ' mesh=', mesh_c, ' n_proj=', n_id
    ENDIF
  END SUBROUTINE load_upf_custom_radials
"""


def patch_text(text: str) -> str:
    if "UPF_CUSTOM_RADIAL_LOADED" in text:
        return text
    for name, blob in (
        ("CALL", CALL_OLD),
        ("header", SUB_OLD),
        ("decl", DECL_OLD),
        ("body", BODY_OLD),
        ("end", "END SUBROUTINE radialpart"),
    ):
        if blob not in text:
            raise RuntimeError(f"radialpart {name} not found")
    if text.count(CALL_OLD) != 1:
        raise RuntimeError("CALL radialpart is not unique")
    if text.count("END SUBROUTINE radialpart") != 1:
        raise RuntimeError("END SUBROUTINE radialpart is not unique")
    text = text.replace(CALL_OLD, CALL_NEW, 1)
    text = text.replace(SUB_OLD, SUB_NEW, 1)
    text = text.replace(DECL_OLD, DECL_NEW, 1)
    text = text.replace(BODY_OLD, BODY_NEW, 1)
    text = text.replace(
        "END SUBROUTINE radialpart",
        LOADER + "END SUBROUTINE radialpart",
        1,
    )
    return text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    args = ap.parse_args()
    raw = args.src.read_text(encoding="utf-8", errors="replace")
    if "\r" in raw:
        raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    out = patch_text(raw)
    args.dst.write_text(out, encoding="utf-8")
    print(f"patched {args.src} -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
