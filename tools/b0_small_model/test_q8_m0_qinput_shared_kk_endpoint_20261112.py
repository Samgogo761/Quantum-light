"""Independent algebra checks for the final endpoint diagnostic. No long runs."""
import json
import unittest
import numpy as np

import q8_m0_qinput_shared_kk_endpoint_20261112 as ep


class EndpointTests(unittest.TestCase):
    def test_centered_covariance_against_explicit_double_sum(self):
        rng=np.random.default_rng(81)
        j=rng.normal(size=(4,3,2))+1j*rng.normal(size=(4,3,2))
        d=rng.normal(size=(4,3))+1j*rng.normal(size=(4,3))
        w=np.array([.1,.2,.3,.4])
        got=ep.ensemble_decomposition(j,d,w)
        prod=np.zeros(2,complex); total=np.zeros(2,complex)
        for a in range(2):
            for q in range(3):
                for r in range(4):
                    total[a]+=w[r]*j[r,q,a]*d[r,q]
                    for s in range(4):
                        prod[a]+=w[r]*w[s]*j[r,q,a]*d[s,q]
        np.testing.assert_allclose(got['product'],prod,atol=1e-14)
        np.testing.assert_allclose(got['total'],total,atol=1e-14)
        np.testing.assert_allclose(got['correlation'],total-prod,atol=1e-14)
        with self.assertRaises(ValueError):
            ep.ensemble_decomposition(j,d,w*2)

    def test_qed_centered_expectation_random_and_product_states(self):
        model=ep.ed.build_models({(2,1)})[(2,1)]
        rng=np.random.default_rng(59)
        v=rng.normal(size=model.dim)+1j*rng.normal(size=model.dim)
        v/=np.linalg.norm(v)
        got=ep.qed_decomposition(model,v)
        np.testing.assert_allclose(got['reconstruction'],0,atol=1e-13)
        np.testing.assert_allclose(got['product_vs_previous_formula'],0,atol=1e-13)
        parts=[rng.normal(size=d)+1j*rng.normal(size=d) for d in (36,36,3)]
        parts=[p/np.linalg.norm(p) for p in parts]
        prod=np.kron(np.kron(parts[0],parts[1]),parts[2])
        np.testing.assert_allclose(ep.qed_decomposition(model,prod)['correlation'],0,atol=1e-13)

    def test_pause_at_cost_probe_does_not_change_propagation(self):
        geo=ep.build_geometry()
        models,grounds,env=ep.cmp.build_bundle(geo,geo.k_cart(ep.sh.K_FRAC))
        node=ep.mp.gauss_hermite_nodes('coherent',5)['nodes'][0]
        ek,emk,_=grounds[1]
        whole=ep.new_state(ek,emk,node); paused=ep.new_state(ek,emk,node)
        ep.advance(whole,models[1],env,4,None,{0,3})
        ep.advance(paused,models[1],env,2,None,{0,3})
        ep.advance(paused,models[1],env,4,None,{0,3})
        for k in ('ek','emk','X','P','rows','times'):
            np.testing.assert_array_equal(whole[k],paused[k])

    def test_odd_and_delta_are_linear_in_each_contribution(self):
        jobs={tag:{'total':np.array([i+1j,i+2j]),'product':np.array([2*i,3*i]),
                   'correlation':np.array([-i+1j,-2*i+2j])}
              for i,(tag,_,_) in enumerate(ep.cmp.JOBS)}
        combos=ep.combinations(jobs)
        for row in combos.values():
            np.testing.assert_allclose(row['total'],row['product']+row['correlation'])
        np.testing.assert_allclose(combos['Delta_J_odd']['total'],
            (jobs['ds_p']['total']-jobs['ds_m']['total']-jobs['coh_p']['total']+jobs['coh_m']['total'])/2)
        zeros={name:{k:np.zeros(2) for k in row} for name,row in combos.items()}
        for row in ep.error_decomposition(combos,zeros).values():
            np.testing.assert_allclose(row['reconstruction'],0)

    def test_complex_archive_serialization(self):
        packed=ep.serializable({'a':np.array([1+2j,3-4j]),'b':np.float64(2)})
        decoded=json.loads(json.dumps(packed,allow_nan=False))
        self.assertEqual(decoded['a'][1],{'re':3.0,'im':-4.0})
        self.assertEqual(ep.CAP,240)
        self.assertEqual(ep.N_STEPS,1285)


if __name__=='__main__':
    unittest.main()
