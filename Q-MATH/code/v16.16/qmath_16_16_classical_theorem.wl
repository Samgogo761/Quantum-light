(* Q-MATH-16.16: optional local Wolfram Language symbolic checks.
   This file is supplied for manual Wolfram verification; it was not executed here. *)
Clear[Ebar, R, p, mQuantum, F2, mClassical];
$Assumptions = Ebar > 0;
F2 = 3 Ebar^2 + Ebar;
R = F2 / Ebar;
p = Ebar/R;
mClassical = p R;
mQuantum = Sqrt[Ebar(Ebar + 1)];
checks = {
  FullSimplify[(1 - p) + p == 1],
  FullSimplify[p R == Ebar],
  FullSimplify[p R^2 == F2],
  FullSimplify[mClassical == Ebar],
  FullSimplify[mQuantum > mClassical],
  FullSimplify[mQuantum/mClassical == Sqrt[(Ebar+1)/Ebar]]
};
Print[checks];
(* For a normalized common emission vector Xi, the leading one-emission
   reduced density operators are F2 Outer[Times, Xi, Conjugate[Xi]]
   for both sources if F2 is matched. *)
