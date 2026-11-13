"""Independent mathematical checks for the corrected R2 implementation."""
import contextlib
import dataclasses
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import solvers as s
import reproduce_all as driver


class MathematicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        s.configure_execution(4)

    def test_assembled_q4_matches_physical_affine_stress_and_boundary_tractions(self):
        # Check connectivity + stiffness + stress recovery, independently of
        # local-node assumptions: exact constant strain and boundary tractions.
        for nx, ny, hx, hy in ((3, 2, 1., 1.), (4, 3, 1.2, .5)):
            E, nu, thickness = 13., .27, 2.
            xy = s._node_coords(nx, ny, nx*hx, ny*hy)
            displacement = np.column_stack([.02*xy[:,0]+.04*xy[:,1],
                                            .03*xy[:,0]-.01*xy[:,1]]).ravel()
            strain = np.array([.02, -.01, .07])
            D = E/(1-nu**2)*np.array([[1.,nu,0.], [nu,1.,0.], [0.,0.,(1-nu)/2]])
            stress = D @ strain
            tensor = np.array([[stress[0], stress[2]], [stress[2], stress[1]]])
            K = s._assemble_K(nx, ny, np.ones((ny,nx)), E0=E, Emin=0., nu=nu,
                              hx=hx, hy=hy, thickness=thickness)
            _, nodes, edof, *_ = s._prepare_fem(nx, ny)
            expected_force = np.zeros((len(xy),2))
            for edge, normal, length in ((nodes[0,:], [0,-1], hx),
                                        (nodes[-1,:], [0,1], hx),
                                        (nodes[:,0], [-1,0], hy),
                                        (nodes[:,-1], [1,0], hy)):
                traction = tensor @ normal
                for first, second in zip(edge[:-1],edge[1:]):
                    expected_force[[first,second]] += thickness*length/2*traction
            np.testing.assert_allclose(K @ displacement, expected_force.ravel(), atol=3e-14)
            self.assertAlmostEqual(displacement @ K @ displacement,
                nx*ny*hx*hy*thickness*(strain @ stress), places=12)
            rotation = np.column_stack([-xy[:,1],xy[:,0]]).ravel()
            np.testing.assert_allclose(K @ rotation, 0., atol=1e-13)
            for dofs in edof:
                np.testing.assert_allclose(s._element_B_center(hx,hy) @ displacement[dofs], strain, atol=1e-15)

    def test_vertical_compliance_is_invariant_to_historical_component_reflection(self):
        nx, ny = 4, 3
        rho = np.linspace(.3,.9,nx*ny).reshape(ny,nx)
        F, fixed = s._l_bracket_load(nx,ny)
        K = s._assemble_K(nx,ny,rho,hx=1.2,hy=.5).toarray()
        signs = np.tile([-1.,1.],len(F)//2)
        old_convention = signs[:,None]*K*signs[None,:]
        free = np.setdiff1d(np.arange(len(F)),fixed)
        def solve(matrix,force):
            u = np.zeros_like(force)
            u[free] = np.linalg.solve(matrix[np.ix_(free,free)],force[free])
            return u
        physical, historical = solve(K,F), solve(old_convention,F)
        np.testing.assert_allclose(physical,signs*historical,atol=1e-9,rtol=1e-11)
        self.assertAlmostEqual(float(F@physical),float(F@historical),places=8)
        # A mixed-direction load exposes why the physical convention matters.
        mixed = F.copy(); mixed[np.flatnonzero(F)[0]-1] = .7
        self.assertGreater(abs(mixed@solve(K,mixed)-mixed@solve(old_convention,mixed)),1.)

    def test_e1_search_streams_separate_runs_cycles_and_purposes(self):
        keys = [(seed,cycle,purpose) for seed in range(7,16)
                for cycle in range(1,13) for purpose in (0,1)]
        seeds = [s._e1_search_seed(*key) for key in keys]
        self.assertEqual(len(set(seeds)),216)
        self.assertEqual(seeds,[s._e1_search_seed(*key) for key in keys])
        self.assertTrue(set(seeds).isdisjoint(s._e1_search_seed(seed,0,2) for seed in range(7,16)))
        problem = s.make_publication_ellipse_problem()
        solver = s.PISCOSolver(problem,dataclasses.replace(s.make_publication_pisco_e1_config(),M_explore=8),
            generator=s.AnalyticEllipseGenerator(problem,s.make_publication_ellipse_box()),cache_enabled=False)
        best = {'a':np.array([60.,30.,10.,10.,0.]),'compliance':1.}
        with patch.object(s,'_multistart_optimize',return_value=torch.empty((0,5))) as starts:
            a = solver._latent_multistart(None,best,[best],run_seed=7,cycle=2)
            first_seed = starts.call_args.kwargs['seed']
            b = solver._latent_multistart(None,best,[best],run_seed=8,cycle=1)
            self.assertNotEqual(first_seed,starts.call_args.kwargs['seed'])
            again = solver._latent_multistart(None,best,[best],run_seed=7,cycle=2)
        self.assertFalse(torch.equal(a,b))
        self.assertTrue(torch.equal(a,again))

    def test_fin_illustration_uses_physical_geometry_and_downward_load(self):
        p = s.FinProblem()
        figure, axis = driver.plt.subplots()
        try:
            driver.PublicationBuilder._fin_background(axis,p)
            driver.PublicationBuilder._fin_support_and_load(axis,p)
            np.testing.assert_allclose(axis.images[0].get_extent(),[0,p.length,0,p.height])
            bearing = axis.patches[0]
            np.testing.assert_allclose(bearing.center,[.13*p.length,.5*p.height])
            self.assertAlmostEqual(bearing.radius,.09*p.length)
            self.assertEqual(axis.get_aspect(),1.)
            F,_ = p.load_and_support()
            self.assertTrue(np.all(F[1::2]<=0) and np.any(F[1::2]<0))
            self.assertGreater(len(axis.texts),0)
            self.assertTrue(all(a.xy[1]<a.get_position()[1] for a in axis.texts))
        finally:
            driver.plt.close(figure)

    def test_rectangular_q4_affine_energy_and_rigid_modes(self):
        # Independent continuum identity: an affine displacement has constant
        # strain, so its energy is area*thickness*strain.T*D*strain.
        for hx, hy, thickness in ((1., 1., 1.), (2/60, 1/36, 1.), (.7, 2.3, 5.)):
            E, nu = 17., .28
            xy = np.array([[0., 0.], [hx, 0.], [hx, hy], [0., hy]])
            u = np.column_stack([.01*xy[:, 0] + .02*xy[:, 1],
                                 -.04*xy[:, 0] + .03*xy[:, 1]]).ravel()
            strain = np.array([.01, .03, -.02])
            D = E/(1-nu**2)*np.array([[1., nu, 0.], [nu, 1., 0.], [0., 0., (1-nu)/2]])
            K = s._lk(E, nu, hx=hx, hy=hy, thickness=thickness)
            self.assertAlmostEqual(u @ K @ u, hx*hy*thickness*(strain @ D @ strain), places=12)
            rotation = np.column_stack([-xy[:, 1], xy[:, 0]]).ravel()
            np.testing.assert_allclose(K @ rotation, 0., atol=1e-13)
            np.testing.assert_allclose(K @ np.tile([1., 0.], 4), 0., atol=1e-13)
            np.testing.assert_allclose(K @ np.tile([0., 1.], 4), 0., atol=1e-13)
            np.testing.assert_allclose(K, K.T, atol=1e-14)

    def test_fin_forward_and_training_use_same_rectangular_stiffness(self):
        p = s.FinProblem(); gen = s.GaussianFinParameterization(p, 35)
        field = gen.decode(np.linspace(.2, .8, 35))
        u, compliance, rho = p.evaluate(field)
        force, fixed = p.load_and_support(); free = np.setdiff1d(np.arange(len(force)), fixed)
        K = s._assemble_K(p.nelx, p.nely, rho, hx=p.length/p.nelx, hy=p.height/p.nely)
        np.testing.assert_allclose((K @ u)[free], force[free], rtol=1e-8, atol=1e-10)
        self.assertAlmostEqual(compliance, force @ u, places=7)
        training = s._training_stiffness(p.nelx, p.nely, rho,
                                         hx=p.length/p.nelx, hy=p.height/p.nely)
        expected = s._csr_to_torch(K)
        self.assertTrue(torch.equal(training.values(), expected.values()))
        square = s._training_stiffness(p.nelx, p.nely, rho)
        self.assertFalse(torch.equal(training.values(), square.values()))

    def test_line_load_resultant_moment_and_thickness(self):
        p = s.make_publication_ellipse_problem()
        for ny in (18, 23, 36):
            force, _ = s._cantilever_load(p.nelx, ny, Lx=p.Lx_mm, Ly=p.Ly_mm,
                load_y1=p.load_y1, load_y2=p.load_y2, q_line=p.q_line, thickness=5.)
            vertical = force[1::2].reshape(ny+1, p.nelx+1, order='F')
            self.assertAlmostEqual(float(vertical.sum()), -1000., places=10)
            self.assertAlmostEqual(float((vertical*np.linspace(0, p.Ly_mm, ny+1)[:, None]).sum()), -30000., places=8)
        one = s._solve_ellipse(dataclasses.replace(p, thickness_mm=1.), [60, 30, 10, 10, 0])
        five = s._solve_ellipse(p, [60, 30, 10, 10, 0])
        np.testing.assert_array_equal(one['F'], five['F'])
        np.testing.assert_allclose(one['U']/5, five['U'], rtol=2e-10, atol=1e-11)
        self.assertAlmostEqual(one['compliance']/5, five['compliance'], places=7)

    def test_projection_jacobian_with_clipping_mask_and_endpoints(self):
        mask = np.array([[True, True, False], [True, True, True]])
        x = torch.tensor([[[-1., .25, 5.], [.6, .85, 2.]]], dtype=torch.float64, requires_grad=True)
        fn = lambda value: s.project_volume_torch(value, mask, .43)
        self.assertTrue(torch.autograd.gradcheck(fn, (x,), eps=1e-6, atol=1e-7, rtol=1e-5))
        y = fn(x)
        self.assertAlmostEqual(float(y.detach()[:, mask].mean()), .43, places=12)
        derivative = torch.autograd.grad(y[:, mask].mean(), x)[0]
        np.testing.assert_allclose(derivative.numpy(), 0., atol=1e-14)
        self.assertEqual(float(y.detach()[0, 0, 2]), 0.)
        for volume in (0., 1.):
            out = s.project_volume_torch(x, mask, volume)
            self.assertTrue(torch.equal(torch.autograd.grad(out.sum(), x)[0], torch.zeros_like(x)))

    def test_fin_decoder_gradient_by_finite_difference(self):
        gen = s.GaussianFinParameterization(s.FinProblem(), 35)
        z = torch.linspace(.2, .8, 35, dtype=torch.float64)[None].requires_grad_(True)
        weights = torch.linspace(-1., 1., 36*60, dtype=torch.float64).reshape(1, 36, 60)
        objective = (gen.decode_torch(z)*weights).sum()
        grad = torch.autograd.grad(objective, z)[0]
        direction = torch.sin(torch.arange(35, dtype=torch.float64))[None]
        eps = 1e-5
        fd = (((gen.decode_torch(z.detach()+eps*direction)-gen.decode_torch(z.detach()-eps*direction))*weights).sum()/(2*eps))
        self.assertAlmostEqual(float((grad*direction).sum()), float(fd), places=7)

    def test_density_filter_gradient_and_physical_oc_volume(self):
        nx, ny = 6, 4; mask = s._l_bracket_mask(nx, ny)
        H, Hs = s._build_filter(nx, ny, 1.5)
        raw = np.linspace(.3, .7, nx*ny).reshape(ny, nx)
        raw = s._initial_filtered_volume(raw, mask, H, Hs, .45)
        force, fixed = s._l_bracket_load(nx, ny)
        def evaluate(x):
            rho = s._density_filter(x, mask, H, Hs)
            _, J, ce, _ = s._fea(nx, ny, rho, fixed, force)
            return J, -3*(1-1e-9)*rho**2*ce
        J, dc = evaluate(raw)
        gradient = s._density_filter_pullback(dc, mask, H, Hs)
        direction = np.random.default_rng(19).normal(size=raw.shape)*mask
        eps = 1e-5
        fd = (evaluate(raw+eps*direction)[0]-evaluate(raw-eps*direction)[0])/(2*eps)
        self.assertAlmostEqual(float((gradient*direction).sum())/fd, 1., places=6)
        new = s._density_oc_update(raw, dc, mask, H, Hs, .45)
        self.assertAlmostEqual(float(s._density_filter(new, mask, H, Hs)[mask].mean()), .45, places=10)
        self.assertLess(evaluate(new)[0], J)

    def test_e1_every_search_query_has_valid_angle_encoding(self):
        class Model(torch.nn.Module):
            def forward(inner, encoded):
                torch.testing.assert_close(encoded[:, 3:].square().sum(dim=1), torch.ones(len(encoded)), rtol=1e-6, atol=1e-6)
                return (encoded[:, 0] + 2*encoded[:, 3] - encoded[:, 4])[:, None]
        p = s.make_publication_ellipse_problem()
        result = s._multistart_optimize(Model(), torch.ones(1), p, s.EllipseBox(),
            [[60., 30., 10., 10., .7]], M=3, steps=8, lr=.4)
        self.assertEqual(tuple(result.shape), (3, 5))
        np.testing.assert_allclose((result[:, 2]*result[:, 3]).numpy(), 100., rtol=1e-6)

    def test_top_k_ties_cannot_be_changed_by_row_order(self):
        a, p = np.arange(16.), np.ones(16)
        for indices in (np.arange(16), np.arange(16)[::-1], np.random.default_rng(4).permutation(16)):
            self.assertIsNone(s.top_k_overlap(a[indices], p[indices], 5))
        self.assertEqual(s.top_k_overlap(a, a, 5), 1.)
        self.assertEqual(s.top_k_overlap(a, -a, 5), 0.)

    def test_manuscript_statistics_use_current_vectors_and_historical_families(self):
        names = [f'{method}_d{dimension}' for dimension in (35,62,135,243)
                 for method in ('pisco','ard_bo','cmaes','de','random')]
        names += ['scalar_surrogate_d35','pisco_no_physics_d35','pisco_progressive','ard_bo_progressive']
        methods = {name:[s.SolverResult(best_compliance=1000.+i*10.+seed*(i+1),
            best_design=None, fem_calls_total=150, trajectory=[], seed=seed, wall_time_s=1., extra={})
            for seed in range(3)] for i,name in enumerate(names)}
        with tempfile.TemporaryDirectory() as tmp:
            out = driver.scientific_outputs(driver.PublicationBuilder(Path(tmp)), {'example3':methods})
        tests = out['statistical_tests']['example3']
        self.assertEqual(len(tests), 22)
        for actual, frozen in zip(tests, driver.FROZEN_E3_COMPARISONS):
            self.assertEqual(actual['family'], frozen['family'])
            self.assertEqual(actual['test'], frozen['test'])
            self.assertGreater(actual['mean_a'], 999.)
        claims = {row['id']:row for row in out['claims']}
        self.assertEqual(len(claims), 94)
        self.assertEqual(claims['e3_pisco_d35']['mean'], 1001.)
        self.assertEqual(claims['e1_pisco']['status'], 'missing_current_evidence')
        self.assertEqual(claims['e2_design_min_feature_size_elements']['status'], 'removed_invalid_interpretation')

    def test_edit_during_execution_cannot_be_stamped_as_executed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'source_probe.py'
            path.write_text('original source')
            s.register_source_guard(path)
            try:
                s._checked_source_digest()
                path.write_text('modified after import')
                with self.assertRaisesRegex(RuntimeError, 'Source changed during execution'):
                    s._checked_source_digest()
            finally:
                s._SOURCE_GUARDS.pop(path.resolve())

    @staticmethod
    def topology_config():
        return dataclasses.replace(s.make_publication_pisco_e2_config(), n_init=4,
            short_simp_iters=1, T_cycles=3, n_top=1, M=2, multistart_steps=2,
            surrogate_steps=2, latent_dim=2, adaptive_pca_max_dim=3,
            adaptive_pca_n_stages=1, adaptive_pca_expand_cycles=(1,),
            local_candidate_pool_size=2, diagnostic_holdout_size=2,
            convergence_patience=None)

    def test_binary_incumbent_and_interrupted_resume(self):
        class Interruption(Exception): pass
        for retain in (False, True):
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                policy = s.TrainingPolicy(retain_optimizer=retain, independent_test_size=0)
                def solver(cache):
                    result = s.BinaryObjectiveHybridPISCOSolver(s.make_publication_lbracket_problem(),
                        self.topology_config(), cache_enabled=cache, cache_dir=Path(tmp)/'cache')
                    result.training_policy = policy
                    return result
                full = solver(False).solve(seed=7)
                original = s.save_checkpoint
                def stop(instance, seed, state):
                    path = original(instance, seed, state)
                    if state['completed_cycle'] == 1: raise Interruption()
                    return path
                with patch.object(s, 'save_checkpoint', side_effect=stop), self.assertRaises(Interruption):
                    solver(True).solve(seed=7)
                resumed = solver(True).solve(seed=7)
                self.assertTrue(np.all((full.best_design==0) | (full.best_design==1)))
                np.testing.assert_array_equal(full.best_design, resumed.best_design)
                self.assertEqual(full.trajectory, resumed.trajectory)
                a, b = full.extra['r2_evidence'], resumed.extra['r2_evidence']
                self.assertEqual(a['predictions'], b['predictions'])
                clean = lambda reports: [{k:v for k,v in r.items() if k!='elapsed_s'} for r in reports]
                self.assertEqual(clean(a['training']), clean(b['training']))
                self.assertEqual(len(full.trajectory), 3)
                self.assertTrue(full.extra['initial_anchor_trajectory'])


if __name__ == '__main__':
    unittest.main()
