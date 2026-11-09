# Stage report formula snippets (LaTeX source)

Use these snippets in a web LaTeX renderer if PowerPoint equation rendering is not good enough. Suggested display style: `\displaystyle`, transparent PNG, dark text.

## Slide 11 - PT/Berry constraint

```latex
\begin{aligned}
\mathcal P &: \Omega(\mathbf k) \mapsto \Omega(-\mathbf k),\\
\mathcal T &: \Omega(\mathbf k) \mapsto -\Omega(-\mathbf k),\\
\mathcal{PT} &: \mathrm{Tr}\,\Omega_{\mathrm{occ}}(\mathbf k)
= -\mathrm{Tr}\,\Omega_{\mathrm{occ}}(\mathbf k).
\end{aligned}
```

```latex
J(t)=J_{\rm intra}(t)+J_{\rm inter}(t),\qquad
J_{\rm anom}(t)\subset J_{\rm inter}(t)\ \text{(to be exported separately)}.
```

## Slide 14 - Bright squeezed vacuum

```latex
|\mathrm{BSV}\rangle
=\operatorname{sech} r\sum_{n=0}^{\infty}
\left(e^{i\phi}\tanh r\right)^n |n,n\rangle .
```

```latex
P(I)=\frac{1}{I_0}\exp\!\left(-\frac{I}{I_0}\right),
\qquad x=I/I_0.
```

```latex
\Delta X_1 < \Delta X_{\rm vac},
\qquad
\Delta X_2 > \Delta X_{\rm vac}.
```

## Slide 15 - BSV average spectrum as intensity average

```latex
\langle S_{\rm BSV}(\omega)\rangle
=\int_0^{\infty} dx\,e^{-x}\,S_{\rm cl}(\omega;I_0 x).
```

```latex
\langle S(\omega)\rangle
\simeq\sum_{j} w_j S_{\rm cl}(\omega;I_j).
```

## Slide 16 - Monte Carlo estimator

```latex
\widehat S_N(\omega)=\frac{1}{N}\sum_{j=1}^{N}S_{\rm cl}(\omega;I_j),
\qquad
\mathrm{SE}[\widehat S_N]=\frac{\sigma_S}{\sqrt N}.
```

## Slide 17 - Deterministic quadrature

```latex
\langle S(\omega)\rangle
\simeq\sum_{j=1}^{16}w_j S_{\rm cl}(\omega;I_j).
```

```latex
\sum_j w_j x_j^p
\simeq
\int_0^{\infty} e^{-x}x^p\,dx
=\Gamma(p+1)=p!.
```

## Slide 19 - p! benchmark

```latex
S_n(I)\propto I^p
\quad\Longrightarrow\quad
R_n\equiv\frac{\langle S_n\rangle_{\rm BSV}}{S_n(I_0)}
=\int_0^{\infty}e^{-x}x^p dx
=\Gamma(p+1)=p!.
```

## Slide 23 - Displacement transformation

```latex
D^{\dagger}(\alpha)aD(\alpha)=\alpha+\delta a.
```

```latex
H' = H_{\rm matter}[E_{\rm cl}(t)]
+H_{\rm int}[\delta a,\delta a^{\dagger}].
```

## Slide 24 - Dyson source operator

```latex
U_I(T)\simeq 1-i\int_0^T dt\,H_{\rm int}(t).
```

```latex
\Lambda_q\propto\int_0^T dt\,e^{i\omega_q t}\,\hat J(t).
```

```latex
\langle a_q\rangle\leftarrow\langle\Lambda_q\rangle,
\qquad
\langle a_q^{\dagger}a_{q'}\rangle\leftarrow
\langle\Lambda_q^{\dagger}\Lambda_{q'}\rangle,
\qquad
\langle a_q a_{q'}\rangle\leftarrow
\langle\Lambda_q\Lambda_{q'}\rangle.
```

## Slide 25 - QRT auxiliary operator

```latex
C^{ab}(t,t')=
\mathrm{Tr}\!\left[
\hat d^a\,\mathcal U(t,t')\bigl(\hat d^b\rho(t')\bigr)
\right],\qquad t\ge t'.
```

```latex
\Lambda(t';t')=\hat d^b\rho(t'),
\qquad
\frac{d\Lambda}{dt}=\mathcal L_{\rm SBE}(t)\Lambda.
```

## Slide 26 - Covariance and N-parity channels

```latex
\Sigma_{ij}=\frac{1}{2}\langle\{\delta R_i,\delta R_j\}\rangle,
\qquad
R=(X_1,P_1,X_2,P_2,\ldots).
```

```latex
C_e=\frac{C(N)+C(-N)}{2},
\qquad
C_o=\frac{C(N)-C(-N)}{2},
\qquad
C_{oo}:\ N\text{-odd}\times N\text{-odd channel}.
```
