#ifndef AMREX_MLMG_3D_K_H_
#define AMREX_MLMG_3D_K_H_

namespace amrex {

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlmg_lin_cc_interp_r2 (Box const& bx, Array4<Real> const& ff,
                            Array4<Real const> const& cc, int nc) noexcept
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            const int kc = k/2;
            const int koff = 2*(k-kc*2)-1;
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = j/2;
                const int joff = 2*(j-jc*2)-1;
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = i/2;
                    const int ioff = 2*(i-ic*2)-1;
                    ff(i,j,k,n) = 0.421875*cc(ic     ,jc     ,kc     ,n)
                        +         0.140625*cc(ic+ioff,jc     ,kc     ,n)
                        +         0.140625*cc(ic     ,jc+joff,kc     ,n)
                        +         0.140625*cc(ic     ,jc     ,kc+koff,n)
                        +         0.046875*cc(ic     ,jc+joff,kc+koff,n)
                        +         0.046875*cc(ic+ioff,jc     ,kc+koff,n)
                        +         0.046875*cc(ic+ioff,jc+joff,kc     ,n)
                        +         0.015625*cc(ic+ioff,jc+joff,kc+koff,n);
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlmg_lin_cc_interp_r4 (Box const& bx, Array4<Real> const& ff,
                            Array4<Real const> const& cc, int nc) noexcept
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            const int kc = k/4;
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = j/4;
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = i/4;
                    ff(i,j,k,n) = cc(ic,jc,kc,n);
                }
            }
        }
    }
}

AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
void mlmg_lin_cc_interp_r8 (Box const& bx, Array4<Real> const& ff,
                            Array4<Real const> const& cc, int nc) noexcept
{
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    /*    //    Array4<Real> ff1(ff);
    Dim3 lo2{lo.x/2,lo.y/2,lo.z/2};
    Dim3 hi2{hi.x/2,hi.y/2,hi.z/2};
    Array4<Real> ff2(ff.p,lo2,hi2,nc);

    Dim3 lo4{lo.x/4,lo.y/4,lo.z/4};
    Dim3 hi4{hi.x/4,hi.y/4,hi.z/4};
    Array4<Real> ff4(ff.p,lo4,hi4,nc);

    for (int n = 0; n < nc; ++n) {
        for (int k = lo4.z; k <= hi4.z; ++k) {
            const int kc = k/2;
            const int koff = 2*(k-kc*2)-1;
            for (int j = lo4.y; j <= hi4.y; ++j) {
                const int jc = j/2;
                const int joff = 2*(j-jc*2)-1;
                AMREX_PRAGMA_SIMD
                for (int i = lo4.x; i <= hi4.x; ++i) {
                    const int ic = i/2;
                    const int ioff = 2*(i-ic*2)-1;
                    ff4(i,j,k,n) = 0.421875*cc(ic     ,jc     ,kc     ,n)
                         +         0.140625*cc(ic+ioff,jc     ,kc     ,n)
                         +         0.140625*cc(ic     ,jc+joff,kc     ,n)
                         +         0.140625*cc(ic     ,jc     ,kc+koff,n)
                         +         0.046875*cc(ic     ,jc+joff,kc+koff,n)
                         +         0.046875*cc(ic+ioff,jc     ,kc+koff,n)
                         +         0.046875*cc(ic+ioff,jc+joff,kc     ,n)
                         +         0.015625*cc(ic+ioff,jc+joff,kc+koff,n);
                }
            }
        }
    }

    for (int n = 0; n < nc; ++n) {
        for (int k = lo2.z; k <= hi2.z; ++k) {
            const int kc = k/2;
            const int koff = 2*(k-kc*2)-1;
            for (int j = lo2.y; j <= hi2.y; ++j) {
                const int jc = j/2;
                const int joff = 2*(j-jc*2)-1;
                AMREX_PRAGMA_SIMD
                for (int i = lo2.x; i <= hi2.x; ++i) {
                    const int ic = i/2;
                    const int ioff = 2*(i-ic*2)-1;
                    ff2(i,j,k,n) = 0.421875*ff4(ic     ,jc     ,kc     ,n)
                         +         0.140625*ff4(ic+ioff,jc     ,kc     ,n)
                         +         0.140625*ff4(ic     ,jc+joff,kc     ,n)
                         +         0.140625*ff4(ic     ,jc     ,kc+koff,n)
                         +         0.046875*ff4(ic     ,jc+joff,kc+koff,n)
                         +         0.046875*ff4(ic+ioff,jc     ,kc+koff,n)
                         +         0.046875*ff4(ic+ioff,jc+joff,kc     ,n)
                         +         0.015625*ff4(ic+ioff,jc+joff,kc+koff,n);
                }
            }
        }
    }

    for (int n = 0; n < nc; ++n) {
        for (int k = lo.z; k <= hi.z; ++k) {
            const int kc = k/2;
            const int koff = 2*(k-kc*2)-1;
            for (int j = lo.y; j <= hi.y; ++j) {
                const int jc = j/2;
                const int joff = 2*(j-jc*2)-1;
                AMREX_PRAGMA_SIMD
                for (int i = lo.x; i <= hi.x; ++i) {
                    const int ic = i/2;
                    const int ioff = 2*(i-ic*2)-1;
                    ff(i,j,k,n) = 0.421875*ff2(ic     ,jc     ,kc     ,n)
                        +         0.140625*ff2(ic+ioff,jc     ,kc     ,n)
                        +         0.140625*ff2(ic     ,jc+joff,kc     ,n)
                        +         0.140625*ff2(ic     ,jc     ,kc+koff,n)
                        +         0.046875*ff2(ic     ,jc+joff,kc+koff,n)
                        +         0.046875*ff2(ic+ioff,jc     ,kc+koff,n)
                        +         0.046875*ff2(ic+ioff,jc+joff,kc     ,n)
                        +         0.015625*ff2(ic+ioff,jc+joff,kc+koff,n);
                }
            }
        }
	}*/







    for (int n = 0; n < nc; ++n) {
      for (int k = lo.z; k <= hi.z; ++k) {
	const int kc = k/8;
	for (int j = lo.y; j <= hi.y; ++j) {
	  const int jc = j/8;
	  AMREX_PRAGMA_SIMD
	    for (int i = lo.x; i <= hi.x; ++i) {
	      const int ic = i/8;
	      ff(i,j,k,n) = cc(ic,jc,kc,n);
	    }
	}
      }
    }
}

}

#endif
