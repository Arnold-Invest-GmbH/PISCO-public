"""R2 numerical and evidence regression tests; no publication claims from tiny runs."""
import contextlib
import copy
import dataclasses
import io
import json
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


class Revision2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_environment_guard_preserves_evidence_on_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = {"probe": {"forward": "aaa"}, "cpu": "reference"}
            driver.guard_reproduction_environment(tmp, original)
            path = Path(tmp) / "reproduction_environment.json"
            saved = path.read_bytes()
            driver.guard_reproduction_environment(tmp, original)
            for modified in ({"probe": {"forward": "bbb"}, "cpu": "reference"},
                             {"probe": {"forward": "aaa"}, "cpu": "different"}):
                with self.assertRaisesRegex(RuntimeError, "environment or numerical probe differs"):
                    driver.guard_reproduction_environment(tmp, modified)
                self.assertEqual(path.read_bytes(), saved)

    def test_environment_probe_repeatability_and_rng_preservation(self):
        s.configure_execution(4)
        before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        first = driver.reproduction_environment()
        self.assertEqual(first, driver.reproduction_environment())
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertTrue(first["native_libraries"])

    def test_packet_training_is_identical_across_worker_counts(self):
        torch.manual_seed(123)
        initial = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Tanh(), torch.nn.Linear(7, 2))
        inputs = torch.randn(11, 3)
        targets = torch.linspace(1., 2., 11)
        matrices = [torch.tensor([[2., -.5], [-.5, 3.]]).to_sparse().coalesce() for _ in range(11)]
        reference = None
        for workers in (1, 4, 12):
            s.configure_execution(workers)
            model = copy.deepcopy(initial)
            audit = s.RunAudit()
            audit.training_policy = s.TrainingPolicy(retain_optimizer=True, validation_fraction=.2,
                normalize_compliance=True, update_steps=2, min_steps=1, patience=2)
            token = s._ACTIVE_AUDIT.set(audit)
            try:
                with s._packet_execution():
                    reports = [s._fit_field(model, inputs, targets, torch.ones(2), matrices,
                        torch.arange(2), 1., steps=4, lr=.001, lam_data=1., lam_phys=.01,
                        physics_batch=5) for _ in range(2)]
            finally:
                s._ACTIVE_AUDIT.reset(token)
            actual = (reports, s._tensor_digest(model.state_dict().values()))
            if reference is None:
                reference = actual
            else:
                self.assertEqual(actual, reference)
        s.configure_execution(4)

    def test_packet_search_is_identical_and_does_not_write_model_gradients(self):
        torch.manual_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
        starts = torch.randn(7, 3)
        reference = None
        for workers in (1, 4, 12):
            s.configure_execution(workers)
            result = s._packet_adam(starts, lambda x: model(x).squeeze(-1), steps=12, lr=.02,
                                   project=lambda x: x.clamp(-1., 1.))
            actual = tuple(s._tensor_digest([v]) for v in result)
            if reference is None:
                reference = actual
            else:
                self.assertEqual(actual, reference)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
        s.configure_execution(4)

    def test_worker_cache_and_environment_provenance(self):
        solver = s.FinPISCOSolver(s.FinProblem(), s.FinSearchConfig(), cache_enabled=False)
        s.configure_execution(1)
        one = solver._cache_payload(2)
        s.configure_execution(12)
        twelve = solver._cache_payload(2)
        self.assertNotEqual(s._payload_hash(one), s._payload_hash(twelve))
        self.assertEqual(one['numerical_execution']['protocol'], s.NUMERICAL_PROTOCOL)
        self.assertEqual(twelve['numerical_execution']['native_threads_per_packet'], 1)
        self.assertIn('scipy', twelve['numerical_execution'])
        s.configure_execution(4)

    def test_old_output_is_rejected_before_overwriting_plan(self):
        import pickle
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = driver.PublicationBuilder(root)
            builder.bundle_path.write_bytes(pickle.dumps({'protocol':'old-protocol','results':{}}))
            plan = root / 'experiment_plan.json'
            plan.write_text('preserve historical evidence')
            with patch.object(sys, 'argv', ['reproduce_all.py','--smoke','--output',str(root)]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    driver.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(plan.read_text(), 'preserve historical evidence')
            self.assertFalse((root / 'runs').exists())

    def test_batched_physics_values_and_gradients(self):
        matrices = [torch.tensor([[2., -1.], [-1., 3.]]).to_sparse().coalesce() for _ in range(3)]
        u = torch.randn(3, 2, requires_grad=True)
        force = torch.tensor([0., 1.])
        actual = s._batched_equilibrium_loss(matrices, u, force, np.arange(2), 1.)
        reference = torch.stack([(torch.sparse.mm(m, u[i, :, None]).ravel()-force).square().sum()
                                 for i, m in enumerate(matrices)]).mean()/(1+1e-12)
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(torch.autograd.grad(actual, u, retain_graph=True)[0],
                                   torch.autograd.grad(reference, u)[0])

    def test_stiffness_cache_preserves_original_assembly_precision(self):
        rho32 = np.random.default_rng(2).uniform(.01, 1., (4, 6)).astype(np.float32)
        rho64 = rho32.astype(np.float64)
        matrices = []
        for rho in (rho32, rho64):
            original = s._csr_to_torch(s._assemble_K(6, 4, rho, penal=3.))
            cached = s._training_stiffness(6, 4, rho, penal=3.)
            self.assertTrue(torch.equal(original.indices(), cached.indices()))
            self.assertTrue(torch.equal(original.values(), cached.values()))
            with patch.object(s, '_assemble_K', side_effect=AssertionError('cache miss')):
                self.assertIs(cached, s._training_stiffness(6, 4, rho.copy(), penal=3.))
            matrices.append(cached)
        # This fixture must expose the old float64 promotion, not just test a
        # density for which both precisions happen to produce the same matrix.
        self.assertFalse(torch.equal(matrices[0].values(), matrices[1].values()))
        rho32[0, 0] = .5
        changed = s._training_stiffness(6, 4, rho32, penal=3.)
        self.assertIsNot(changed, matrices[0])
        original = s._csr_to_torch(s._assemble_K(6, 4, rho32, penal=3.))
        self.assertTrue(torch.equal(original.values(), changed.values()))

    def test_zero_physics_and_retained_optimizer(self):
        model = torch.nn.Linear(2, 2)
        inputs = torch.randn(8, 2)
        audit = s.RunAudit()
        audit.training_policy = s.TrainingPolicy(retain_optimizer=True, validation_fraction=.25,
                                                update_steps=1, independent_test_size=0)
        token = s._ACTIVE_AUDIT.set(audit)
        try:
            with patch.object(s, '_block_stiffness', side_effect=AssertionError('physics work at lambda=0')):
                first = s._fit_field(model, inputs, torch.ones(8), torch.ones(2), [], None, 1.,
                                     steps=2, lr=.001, lam_data=1., lam_phys=0.)
                second = s._fit_field(model, inputs, torch.ones(8), torch.ones(2), [], None, 1.,
                                      steps=2, lr=.001, lam_data=1., lam_phys=0.)
            self.assertEqual(first['training_size'], 6)
            self.assertEqual(second['steps_executed'], 1)
            self.assertEqual(float(next(iter(model._r2_optimizer['state'].values()))['step']), 3.)
        finally:
            s._ACTIVE_AUDIT.reset(token)

    def test_frozen_weights_preserve_input_gradient(self):
        model = torch.nn.Linear(2, 1)
        def search(model):
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            x = torch.ones(1, 2, requires_grad=True)
            model(x).sum().backward()
            self.assertIsNotNone(x.grad)
        s._audited_method('candidate_search', search)(model)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_mesh_transfer_and_invalid_continuous_input(self):
        problem = dataclasses.replace(s.make_publication_lbracket_problem(), nelx=12, nely=12)
        mask = s._l_bracket_mask(12, 12).astype(float)
        result = s.BinaryMeshRefinementSolver(problem, s.MeshRefinementConfig(), mask, cache_enabled=False).solve(seed=0)
        rows = result.extra['mesh_refinement']
        self.assertEqual(rows[0]['volume_fraction'], rows[1]['volume_fraction'])
        self.assertEqual(rows[0]['total_vertical_force'], rows[1]['total_vertical_force'])
        self.assertEqual(result.extra['r2_evidence']['actual_recorded_fem_calls'], 2)
        with self.assertRaises(ValueError):
            s.BinaryMeshRefinementSolver(problem, s.MeshRefinementConfig(), mask*.5)

    def test_ellipse_and_documentation(self):
        problem = s.make_publication_ellipse_problem()
        config = dataclasses.replace(s.make_publication_pisco_e1_config(), n_init=4, T_cycles=1,
            n_top=1, M=2, M_explore=2, multistart_steps=2, surrogate_steps=2,
            surrogate_hidden=8, surrogate_depth=1)
        solver = s.PISCOSolver(problem, config,
            generator=s.AnalyticEllipseGenerator(problem, s.make_publication_ellipse_box()), cache_enabled=False)
        solver.training_policy = s.TrainingPolicy(independent_test_size=2)
        result = solver.solve(seed=7)
        audit = result.extra['r2_evidence']
        self.assertEqual(len(audit['predictions']), 1)
        self.assertEqual(len(audit['test_predictions']), 2)
        self.assertTrue(audit['incumbents'])
        self.assertEqual(result.fem_calls_total, 5)
        self.assertEqual(len(audit['fem_events']), 7)
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            builder = driver.PublicationBuilder(Path(tmp))
            builder.save_results({'example1': {'pisco': [result]}})
            status = driver.write_r2_documentation(builder, builder.load_results())
            self.assertFalse(status['full_publication_complete'])
            self.assertEqual(driver.verify_r2(builder), 1)
            root = Path(tmp)/'documentation'
            manifest = json.loads((root/'artifact_manifest.json').read_text())
            self.assertIn('example1/pisco/seed_7/diagnostics.png', manifest)
            altered = dataclasses.replace(result, seed=8, extra={**result.extra,
                'r2_evidence': {**audit, 'configuration': {**audit['configuration'], 'source_sha256':'wrong'}}})
            with self.assertRaises(RuntimeError):
                builder.save_results({'example1': {'pisco': [altered]}})

    def test_fin_field_and_scalar(self):
        config = s.FinSearchConfig(n_init=3, budget=4, n_top=1, surrogate_steps=1,
                                  multistart_count=2, multistart_steps=1)
        for cls in (s.FinPISCOSolver, s.FinScalarPISCOSolver):
            with self.subTest(solver=cls.__name__):
                solver = cls(s.FinProblem(), config, cache_enabled=False)
                solver.training_policy = s.TrainingPolicy(independent_test_size=2)
                result = solver.solve(seed=7)
                self.assertEqual(result.fem_calls_total, 4)
                self.assertEqual(len(result.extra['r2_evidence']['test_predictions']), 2)
                self.assertTrue(result.extra['r2_evidence']['training'])

    def test_topology_variants_and_checkpoint_resume(self):
        config = dataclasses.replace(s.make_publication_pisco_e2_config(), n_init=4,
            short_simp_iters=1, T_cycles=1, n_top=1, M=2, multistart_steps=1,
            surrogate_steps=1, latent_dim=2, adaptive_pca_max_dim=3, adaptive_pca_n_stages=1,
            adaptive_pca_expand_cycles=(1,), local_candidate_pool_size=2, diagnostic_holdout_size=2)
        for cls in (s.RevisionPISCOSolverTopology, s.BinaryObjectiveHybridPISCOSolver, s.MatchedScalarSurrogateSolver):
            with self.subTest(solver=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                kwargs = dict(cache_dir=Path(tmp)/'_solver_cache')
                solver = cls(s.make_publication_lbracket_problem(), config, **kwargs)
                solver.training_policy = s.TrainingPolicy(retain_optimizer=True, independent_test_size=0)
                first = solver.solve(seed=7)
                self.assertEqual(len(first.extra['r2_evidence']['predictions']), 1)
                solver._cache_path(7).unlink()
                resumed = cls(s.make_publication_lbracket_problem(), config, **kwargs)
                resumed.training_policy = solver.training_policy
                second = resumed.solve(seed=7)
                np.testing.assert_array_equal(first.best_design, second.best_design)
                self.assertEqual(first.best_compliance, second.best_compliance)
                self.assertTrue(second.extra['r2_evidence']['resumed'])
                self.assertEqual(len(first.extra['r2_evidence']['training']), len(second.extra['r2_evidence']['training']))

    def test_live_heartbeat_and_progress_snapshot(self):
        import threading
        emitted = threading.Event()
        class ObservedProgress(s.LiveProgress):
            def emit(self, state, **kwargs):
                super().emit(state, **kwargs)
                if threading.current_thread() is self.thread:
                    emitted.set()
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            live = ObservedProgress(tmp,"example1","pisco",9,batch=(3,5),durations=(10.,12.),interval=.01)
            audit = s.RunAudit()
            audit.stage = "surrogate_training"
            audit.progress = {"stage":audit.stage,"label":"training update 2","completed":3,"total":10}
            live.audit = audit
            with live:
                self.assertTrue(emitted.wait(2), "Heartbeat was not emitted during a blocked operation")
                row=json.loads((Path(tmp)/"progress.json").read_text())
                self.assertEqual(row["progress"]["completed"],3)
                self.assertIsNotNone(row["remaining_method_eta_s"])
                live.emit("completed")
            self.assertFalse(live.thread.is_alive())
            self.assertIsNone(s._LIVE_PROGRESS.get())
            rows=[json.loads(line) for line in (Path(tmp)/"progress.jsonl").read_text().splitlines()]
            self.assertEqual(rows[-1]["state"],"completed")

    def test_live_failure_and_unavailable_eta(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            live=s.LiveProgress(tmp,"example1","pisco",7,batch=(1,9),interval=.01)
            with self.assertRaisesRegex(ValueError,"intentional"):
                with live:
                    raise ValueError("intentional test failure")
            row=json.loads((Path(tmp)/"progress.json").read_text())
            self.assertEqual(row["state"],"failed")
            self.assertIsNone(row["remaining_method_eta_s"])
            self.assertFalse(live.thread.is_alive())
            self.assertIsNone(s._LIVE_PROGRESS.get())

    def test_result_store_cache_excluded_from_eta(self):
        class TinySolver(s.BaseSolver):
            def _compute(self, *, seed):
                return s.SolverResult(1.,np.ones((1,1)),0,[],seed,0.,{})
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            store=s.ResultStore(Path(tmp)/"runs")
            store.solve("test","tiny",TinySolver(None,None),7)
            store.solve("test","tiny",TinySolver(None,None),7)
            self.assertEqual(len(store._fresh_durations[("test","tiny")]),1)
            row=json.loads((Path(tmp)/"progress.json").read_text())
            self.assertEqual(row["state"],"cached")
            self.assertTrue(row["cached"])

    def test_de_fixed_budget_all_publication_seeds(self):
        config=dataclasses.replace(s.make_publication_baseline_configs()[0],
            maxiter=3,polish=False,tol=0.0,atol=0.0)
        for seed in s.PUBLICATION_SEEDS:
            with self.subTest(seed=seed):
                result=s.DESolver(s.make_publication_ellipse_problem(),config,cache_enabled=False).solve(seed=seed)
                self.assertEqual(result.fem_calls_total,160)
                checkpoint=s.result_at_fem_checkpoint(result,156)
                self.assertEqual(checkpoint.fem_calls_total,156)

    def test_declarative_experiment_matrix(self):
        # Exercise orchestration without expensive solver calls.
        class Recorder:
            def solve_many(self, group, method, seeds, factory):
                return [s.SolverResult(1., None, 160, [(156, 1.)], seed, 0., {}) for seed in seeds]
            def solve(self, group, method, solver, seed):
                return s.SolverResult(1., None, 160, [(156, 1.)], seed, 0., {})
        suite = s.PublicationExperimentSuite('/unused')
        suite.store = Recorder()
        result = {'example1':suite.example1(), 'example2':suite.example2(), 'example3':suite.example3()}
        observed = {f'{family}/{method}':[r.seed for r in runs]
                    for family, methods in result.items() for method,runs in methods.items()}
        plan = s.expected_publication_experiments()
        for key,seeds in plan.items():
            if not key.startswith('mesh_refinement/'):
                self.assertEqual(observed[key], seeds, key)
        self.assertEqual(set(observed), {k for k in plan if not k.startswith('mesh_refinement/')})


if __name__ == '__main__':
    unittest.main()
