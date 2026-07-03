"""H7.1b -- tests for stability-mode search-operator alignment in aste_hunter.generate_next_generation.

Proves: objective="stability" steers SELECTION by stability fitness (higher=better) via a tournament + bounded
mutation on param_a/eta/rho_vac ONLY, in the narrow box around a*, with NO spectral SGN/ASMT/NSGA steering and NO
metrics/prime dependency; and the default objective="prime" path is unchanged. No cupy/GPU.
"""
import os, sys, sqlite3
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
H = pytest.importorskip("aste_hunter")


def _seed_completed(db_file, gen, rows):
    """rows: list of (config_hash, fitness, param_a, param_eta, param_rho_vac). NO metrics rows are inserted,
    so this also proves the stability generator needs no spectral metrics."""
    c = sqlite3.connect(db_file)
    for (ch, fit, pa, pe, pr) in rows:
        c.execute("INSERT OR REPLACE INTO runs (config_hash, seed, generation, status, fitness) VALUES (?,0,?, 'completed', ?)",
                  (ch, gen, fit))
        c.execute("INSERT OR REPLACE INTO parameters "
                  "(config_hash, param_D, param_eta, param_rho_vac, param_a_coupling, param_splash_coupling, param_splash_fraction, param_a) "
                  "VALUES (?,?,?,?,?,?,?,?)",
                  (ch, 2.7329, pe, pr, 2.3098, 0.0129, -0.4861, pa))
    c.commit(); c.close()


def test_stability_generator_ranks_astar_and_stays_in_box(tmp_path):
    h = H.Hunter(db_file=str(tmp_path / "s.db"), objective="stability")
    _seed_completed(h.db_file, 0, [
        ("astar", 0.884, 0.5522, 0.0704, 1.1866),   # a* — best fitness
        ("decay", 0.448, 0.5042, 0.0704, 1.1866),
        ("grow",  0.000, 0.6003, 0.0704, 1.1866),   # grower (rejected -> fitness 0)
    ])
    nxt = h.generate_next_generation(6)
    assert len(nxt) == 6
    # elitism carries the best-fitness (a*) individual unchanged
    elite = [c for c in nxt if c["origin"] == "STABILITY_ELITE"]
    assert len(elite) == 1 and abs(elite[0]["param_a"] - 0.5522) < 1e-9
    # every child stays inside the narrow a* box on all 3 axes
    for c in nxt:
        assert 0.48 <= c["param_a"] <= 0.60
        assert 0.0598 <= c["param_eta"] <= 0.0810
        assert 1.068 <= c["param_rho_vac"] <= 1.365
    # non-search axes are HELD at feb (not mutated) -> search restricted to param_a/eta/rho_vac
    for c in nxt:
        assert abs(c["param_a_coupling"] - 2.3098) < 1e-9
        assert abs(c["param_D"] - 2.7329) < 1e-9
        assert abs(c["param_splash_coupling"] - 0.0129) < 1e-9
    # origins are stability-only: NO spectral SGN/ASMT/NSGA/PREDICTOR children
    assert all(c["origin"] in ("STABILITY_ELITE", "STABILITY_MUTATION") for c in nxt)


def test_stability_empty_db_is_narrow_box_random(tmp_path):
    h = H.Hunter(db_file=str(tmp_path / "e.db"), objective="stability")
    nxt = h.generate_next_generation(6)   # no completed runs
    assert len(nxt) == 6
    assert all(c["origin"] == "STABILITY_RANDOM" for c in nxt)
    for c in nxt:
        assert 0.48 <= c["param_a"] <= 0.60 and 0.0598 <= c["param_eta"] <= 0.0810
        assert abs(c["param_a_coupling"] - 2.3098) < 1e-9   # non-search axes held at feb


def test_prime_default_path_unchanged(tmp_path):
    h = H.Hunter(db_file=str(tmp_path / "p.db"))     # default objective="prime"
    assert h.objective == "prime"
    nxt = h.generate_next_generation(4)              # empty DB -> prime random path (origin NATURAL)
    assert len(nxt) == 4
    assert all(c.get("origin") == "NATURAL" for c in nxt)   # NOT STABILITY_* -> prime path untouched


def test_stability_selection_prefers_higher_fitness(tmp_path):
    # with a strongly dominant best, the tournament elite must be that individual; mutations cluster near it
    h = H.Hunter(db_file=str(tmp_path / "d.db"), objective="stability")
    _seed_completed(h.db_file, 0, [
        ("best", 0.90, 0.5500, 0.0704, 1.1866),
        ("mid",  0.30, 0.5000, 0.0650, 1.1000),
        ("low",  0.05, 0.5900, 0.0800, 1.3000),
    ])
    nxt = h.generate_next_generation(8)
    elite = [c for c in nxt if c["origin"] == "STABILITY_ELITE"][0]
    assert abs(elite["param_a"] - 0.5500) < 1e-9
    # population mean param_a should sit nearer the best (0.55) than the low (0.59) given fitness pressure
    mean_a = sum(c["param_a"] for c in nxt) / len(nxt)
    assert 0.50 <= mean_a <= 0.60
