(* ::Package:: *)
(* M0 independent Wolfram Language rebuild from source-derived geometry JSON.
   Run: wolframscript -file M0_Wolfram_independent_validation.wl
   Source: user supplied q8_m0_model.py + q8_tperp_static_audit.py + CrI3_scf_soc_a1h_clean.in.
   No Python import and no GUI dependency. Mathematica itself was not available to
   this assistant, so PLEASE execute this script locally and inspect pass/fail.
*)
ClearAll["Global`*"];
base = If[StringQ[$InputFileName] && StringLength[$InputFileName] > 0,
          DirectoryName[ExpandFileName[$InputFileName]], Directory[]];
geomFile = FileNameJoin[{base, "M0_geometry_reference_uploaded.json"}];
If[!FileExistsQ[geomFile], Print["MISSING: ", geomFile]; Exit[2]];
dat = Import[geomFile, "RawJSON"];
geo = dat["geometry"]; par = dat["parameters"];

lat = geo["lattice_ang"]; recip = geo["reciprocal_2d"];
nn = (Take[#, 2] &) /@ geo["nn_deltas_cart"];
nnn = geo["nnn_terms"]; bonds = geo["interlayer_bonds"];
th = par["t_h"]; lso = par["lam_so"]; delta = par["delta"];
s0 = IdentityMatrix[2]; sx = PauliMatrix[1]; sy = PauliMatrix[2]; sz = PauliMatrix[3];
p = KroneckerProduct[sx, sx, s0];
ut = KroneckerProduct[s0, s0, I sy]; upt = p.ut;
spin8 = KroneckerProduct[s0,s0,sz]; id8 = IdentityMatrix[8];

fM[k_] := Total[(Exp[I Dot[k,#]] &) /@ nn];
f1M[k_, a_] := Total[(I #[[a]] Exp[I Dot[k,#]] &) /@ nn];
f2M[k_, a_, b_] := Total[(-#[[a]] #[[b]] Exp[I Dot[k,#]] &) /@ nn];
gM[k_] := 2 Total[(#[[1]] Sin[Dot[k, Take[#[[2]],2]]] &) /@ nnn];
g1M[k_,a_] := 2 Total[(#[[1]] #[[2,a]] Cos[Dot[k, Take[#[[2]],2]]] &) /@ nnn];
g2M[k_,a_,b_] := -2 Total[(#[[1]] #[[2,a]] #[[2,b]] Sin[Dot[k, Take[#[[2]],2]]] &) /@ nnn];

(* order=0 ordinary hopping, 1 first derivative, 2 second derivative.
   Indices a,b are Cartesian in-plane indices 1,2. *)
interTerm[k_, bond_, order_, a_, b_] := Module[{dv, ii, jj, coefficient},
  dv = Take[bond["delta_cart"],2];
  ii = If[StringTake[bond["from"],-1] == "A",1,2];
  jj = If[StringTake[bond["to"],-1] == "A",1,2];
  coefficient = Switch[order, 0, 1, 1, I dv[[a]], 2, -dv[[a]] dv[[b]]];
  par[bond["amplitude"]] Exp[I Dot[k,dv]] coefficient
      Outer[Times, UnitVector[2,ii], UnitVector[2,jj]]
];
tM[k_] := Total[(interTerm[k,#,0,1,1] &) /@ bonds];
t1M[k_,a_] := Total[(interTerm[k,#,1,a,1] &) /@ bonds];
t2M[k_,a_,b_] := Total[(interTerm[k,#,2,a,b] &) /@ bonds];

layerBlock[fval_,gval_,neel_,eta_] :=
  th KroneckerProduct[Re[fval] sx - Im[fval] sy, s0]
 + lso gval KroneckerProduct[sz,sz]
 + neel eta delta KroneckerProduct[s0,sz];

hM[k_,n_] := Module[{tt = KroneckerProduct[tM[k],s0]},
  ArrayFlatten[{{layerBlock[fM[k],gM[k],n,1],tt},
                {ConjugateTranspose[tt], layerBlock[fM[k],gM[k],n,-1]}}]
];
(* Equivalent to the original analytic first and second momentum vertices.
   Input coupling is k -> k+e A/hbar, e=hbar=1. The minus sign on j
   follows current = - derivative of h with respect to k. *)
d1M[k_,a_] := Module[{hh,tt},
  hh = layerBlock[f1M[k,a],g1M[k,a],0,1];
  tt = KroneckerProduct[t1M[k,a],s0];
  ArrayFlatten[{{hh,tt},{ConjugateTranspose[tt],hh}}]
];
d2M[k_,a_,b_] := Module[{hh,tt},
  hh = layerBlock[f2M[k,a,b],g2M[k,a,b],0,1];
  tt = KroneckerProduct[t2M[k,a,b],s0];
  ArrayFlatten[{{hh,tt},{ConjugateTranspose[tt],hh}}]
];

orbSites = {"tA","tA","tB","tB","bA","bA","bB","bB"};
taus = (Dot[geo["sites_paired_frac"][#],lat] &) /@ orbSites;
sew[G_] := DiagonalMatrix[(Exp[I Dot[Take[#,2],G]] &) /@ taus];
kFrac[kf_] := Dot[kf,recip];

names = {"Hermiticity", "P", "T", "PT", "PT2", "SpinZ", "CurrentP",
 "CurrentT", "SewH", "SewJ", "Derivative1", "Derivative2", "Kramers"};
res = AssociationThread[names,ConstantArray[0.,Length[names]]];
update[key_, value_] := AssociateTo[res,key -> Max[res[key],N[value]]];
update["PT2", Norm[upt.Conjugate[upt] + id8, "Frobenius"]];
fracPoints = {{0.,0.},{1/3.,1/3.},{0.21,0.13},{0.44,0.37},{0.1,0.87},{0.5,0.5},{-0.33,0.72}};
Glist = {recip[[1]],recip[[2]]};
eps1 = 1.*^-5; eps2 = 2.*^-5;

Do[
 Module[{k,h,hm,j,jm,eval,gv,uu,ee,step,derNum,derNum2},
  k = kFrac[kf]; h = hM[k,n]; hm = hM[-k,-n];
  j = Table[-d1M[k,a],{a,1,2}];
  jm = Table[-d1M[-k,a],{a,1,2}];
  update["Hermiticity",Norm[h-ConjugateTranspose[h],"Frobenius"]];
  update["P",Norm[p.h.ConjugateTranspose[p]-hm,"Frobenius"]];
  update["T",Norm[ut.Conjugate[h].ConjugateTranspose[ut]-hm,"Frobenius"]];
  update["PT",Norm[upt.Conjugate[h].ConjugateTranspose[upt]-h,"Frobenius"]];
  update["SpinZ",Norm[h.spin8-spin8.h,"Frobenius"]];
  eval=Sort[Re[Eigenvalues[h]]];
  update["Kramers",Max[(Abs[#[[1]]-#[[2]]]&) /@ Partition[eval,2]]];
  Do[
    update["CurrentP",Norm[p.j[[a]].ConjugateTranspose[p]+jm[[a]],"Frobenius"]];
    update["CurrentT",Norm[ut.Conjugate[j[[a]]].ConjugateTranspose[ut]+jm[[a]],"Frobenius"]];
    step=eps1 UnitVector[2,a];
    derNum=(hM[k+step,n]-hM[k-step,n])/(2 eps1);
    update["Derivative1",Norm[derNum-d1M[k,a],"Frobenius"]];
    Do[
      step=eps2 UnitVector[2,b];
      derNum2=(d1M[k+step,a]-d1M[k-step,a])/(2 eps2);
      update["Derivative2",Norm[derNum2-d2M[k,a,b],"Frobenius"]];,
    {b,1,2}];,
  {a,1,2}];
  Do[
    gv=sew[G];
    update["SewH",Norm[hM[k+G,n]-ConjugateTranspose[gv].h.gv,"Frobenius"]];
    Do[
      update["SewJ",Norm[d1M[k+G,a]-ConjugateTranspose[gv].d1M[k,a].gv,"Frobenius"]];,
    {a,1,2}];,
  {G,Glist}];
 ];,
{n,{-1,1}},{kf,fracPoints}];

(* Exterior-power comparison at exactly the same illustrative field histories
   and k used by M0_influence_exterior_validation.py. *)
kTest = kFrac[{0.21,0.32}];
ePol = Take[geo["a2_cart"],2]; ePol = ePol/Norm[ePol];
aPlus = {0.0,0.048,-0.025,0.033};
aMinus = {0.0,0.037,-0.025,0.041};
dt = {0.18,0.15,0.20,0.12};
energiesAndVectors = Eigensystem[hM[kTest,1]];
order = Ordering[Re[energiesAndVectors[[1]]],4];
vOcc = Transpose[energiesAndVectors[[2,order]]];
proj = vOcc.ConjugateTranspose[vOcc];
uFor[amps_] := Fold[
 (MatrixExp[-I #2[[2]] hM[kTest+#2[[1]] ePol,1]].#1)&,
 id8,Transpose[{amps,dt}]];
up=uFor[aPlus]; um=uFor[aMinus];
w=ConjugateTranspose[um].up;
f4=Det[ConjugateTranspose[vOcc].w.vOcc];
f8=Det[id8-proj+proj.w];
comb = Subsets[Range[8],{4}];
psiPlus=up.vOcc; psiMinus=um.vOcc;
wPlus=(Det[psiPlus[[#,All]]]&) /@ comb;
wMinus=(Det[psiMinus[[#,All]]]&) /@ comb;
f70=Conjugate[wMinus].wPlus;
(* Independent inversion covariance of the full two-branch electron influence.
   Inverts BOTH histories, k and the Neel domain. *)
esInv=Eigensystem[hM[-kTest,-1]];
iOcc=Ordering[Re[esInv[[1]]],4];
vInv=Transpose[esInv[[2,iOcc]]];
uInverted[amps_] := Fold[
 (MatrixExp[-I #2[[2]] hM[-kTest+#2[[1]] ePol,-1]].#1)&,
 id8,Transpose[{amps,dt}]];
upInv=uInverted[-aPlus]; umInv=uInverted[-aMinus];
fInv=Det[ConjugateTranspose[vInv].ConjugateTranspose[umInv].upInv.vInv];
influenceChecks = <|
  "dimExterior"->Length[comb],
  "det4ReIm"->{Re[f4],Im[f4]},
  "errDet8"->Abs[f8-f4],
  "errWedge70"->Abs[f70-f4],
  "inversionInfluenceError"->Abs[fInv-f4],
  "diagonalNormalization"->Abs[Det[ConjugateTranspose[vOcc].ConjugateTranspose[up].up.vOcc]-1],
  "unitarityPlus"->Norm[ConjugateTranspose[up].up-id8,"Frobenius"]
|>;

Print["M0 Wolfram independent static/vertex checks:"];
Print[res];
Print["Exact free-electron influence checks:"];
Print[influenceChecks];

expectedFile=FileNameJoin[{base,"M0_influence_exterior_results.json"}];
If[FileExistsQ[expectedFile],
  expected=Import[expectedFile,"RawJSON"];
  ref=expected["branch_overlap"]["det4"];
  influenceChecks["PythonCrossCheck"] = Abs[f4-(ref[[1]]+I ref[[2]])];
  Print["Independent Python 8x8 vs Mathematica 8x8 overlap gap: ",influenceChecks["PythonCrossCheck"]];
];

output=<|"model"->"Q_M0_eightband_v1",
  "source"->"Independent Wolfram equations using source-generated geometry JSON",
  "staticResiduals"->res,"influenceChecks"->influenceChecks|>;
Export[FileNameJoin[{base,"M0_Wolfram_validation_results.json"}],output,"RawJSON"];
threshold=1.*^-6;
maxError=Max[Join[Values[res],
  {influenceChecks["errDet8"],influenceChecks["errWedge70"],
   influenceChecks["diagonalNormalization"],influenceChecks["unitarityPlus"],
   influenceChecks["inversionInfluenceError"]},
  If[KeyExistsQ[influenceChecks,"PythonCrossCheck"],{influenceChecks["PythonCrossCheck"]},{}]]];
If[!NumericQ[maxError] || maxError>threshold,
   Print["FAILED: largest check = ",maxError]; Exit[1],
   Print["PASSED: largest check = ",maxError]
];
