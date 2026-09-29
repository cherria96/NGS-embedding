"""Item 18.2: mathematical scale invariance. After zero handling,
multiplying a domain's positive vector by a sample-specific positive factor
leaves its log-contrast score unchanged -- but re-adding a FRESH pseudocount
to arbitrarily rescaled raw counts is not required to reproduce that."""
import numpy as np


def log_contrast_score(tilde_c, beta):
    q = tilde_c / tilde_c.sum()
    return float(np.log(q) @ beta)


def test_positive_vector_rescaling_invariance():
    rng = np.random.default_rng(1)
    p = 6
    raw = rng.integers(0, 50, size=p).astype(float)
    delta = 0.5
    tilde_c = raw + delta
    beta = rng.normal(size=p)
    beta -= beta.mean()  # zero-sum

    score_original = log_contrast_score(tilde_c, beta)
    for k in (0.001, 1.0, 7.3, 1000.0):
        rescaled = k * tilde_c  # scale the POSITIVE (already zero-handled) vector
        score_rescaled = log_contrast_score(rescaled, beta)
        assert abs(score_original - score_rescaled) < 1e-10


def test_fresh_pseudocount_after_rescaling_raw_counts_is_not_required_to_match():
    """Scaling RAW counts and then adding a fresh pseudocount is a different
    operation from scaling the already-pseudocounted positive vector; the
    spec explicitly does not require invariance in that case."""
    rng = np.random.default_rng(2)
    p = 6
    raw = rng.integers(0, 50, size=p).astype(float)
    delta = 0.5
    beta = rng.normal(size=p)
    beta -= beta.mean()

    tilde_c_original = raw + delta
    score_original = log_contrast_score(tilde_c_original, beta)

    k = 37.0
    tilde_c_fresh_pseudocount = (k * raw) + delta  # pseudocount re-added AFTER rescaling raw counts
    score_fresh = log_contrast_score(tilde_c_fresh_pseudocount, beta)

    # These generally differ because delta does not scale with k (unless
    # delta == 0, or raw is degenerate); this is the expected, documented
    # behavior -- not a bug to "fix" by requiring equality here.
    assert abs(score_original - score_fresh) > 1e-6
