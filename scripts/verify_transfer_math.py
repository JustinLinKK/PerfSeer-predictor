"""Numerical checks of the proposal, not PerfSeer accuracy experiments."""
import json
import numpy as np

rng = np.random.default_rng(29)
checks = []

def check(name, ok, detail):
    assert bool(ok), (name, detail)
    checks.append({"name": name, "passed": True, "detail": detail})

def ridge(phi, residual, weights, lam):
    a = phi.T @ (weights[:, None] * phi) + lam * np.eye(phi.shape[1])
    b = phi.T @ (weights * residual)
    return np.linalg.solve(a, b), a, b

n, d = 128, 9
phi = np.c_[np.ones(n), rng.normal(size=(n, d-1))]
weights = rng.uniform(.5, 1.5, n)
weights /= weights.sum()
truth = rng.normal(scale=.08, size=d)
residual = phi @ truth
w, a, b = ridge(phi, residual, weights, 1e-8)
gradient = 2 * (a @ w - b)
check("ridge_stationarity", np.linalg.norm(gradient) < 1e-10,
      float(np.linalg.norm(gradient)))
check("ridge_strict_convexity", np.linalg.eigvalsh(2*a).min() > 0,
      float(np.linalg.eigvalsh(2*a).min()))

phiq = np.c_[np.ones(64), rng.normal(size=(64,d-1))]
source_time = np.exp(rng.normal(size=64))
true_time = source_time * np.exp(phiq @ truth)
prediction = source_time * np.exp(phiq @ w)
relative = np.abs(prediction - true_time) / true_time
check("representable_time_transfer_heldout", relative.max() < 1e-6,
      {"max_relative_error": float(relative.max()), "query_workloads":64})

zero, _, _ = ridge(phi, np.zeros(n), weights, 1e-3)
check("zero_residual_identity", np.array_equal(zero, np.zeros(d)) and
      np.array_equal(source_time*np.exp(phiq@zero), source_time), "exact identity")

degenerate = np.c_[np.ones(8), np.ones(8), np.zeros(8)]
wdeg, adeg, _ = ridge(degenerate, np.ones(8), np.ones(8)/8, .01)
check("rank_deficient_design_regularized", np.isfinite(wdeg).all() and
      np.linalg.eigvalsh(adeg).min() > 0, float(np.linalg.cond(adeg)))

true_source, predicted_source, true_target = 100., 110., 120.
wrong = predicted_source * np.exp(np.log(true_target/true_source))
right = predicted_source * np.exp(np.log(true_target/predicted_source))
check("source_error_propagation_counterexample", abs(wrong-132.)<1e-10 and
      abs(right-120.)<1e-10, {"paired_truth_ratio_prediction":wrong,
                             "inference_matched_residual_prediction":right})

mem_truth = rng.normal(scale=.07,size=d)
wm, _, _ = ridge(phi, phi@mem_truth, weights, 1e-8)
baseline = np.exp(rng.normal(7,.5,64))
scale = np.maximum(baseline,64.)
target = baseline+scale*(phiq@mem_truth)
raw = baseline+scale*(phiq@wm)
check("representable_memory_transfer_heldout", target.min()>0 and
      np.max(np.abs(raw-target)/target)<1e-6,
      {"max_relative_error":float(np.max(np.abs(raw-target)/target))})

raw_values = rng.normal(0,5,1000)
positive_truth = rng.uniform(0,10,1000)
projected = np.maximum(raw_values,0)
check("nonnegative_projection_error", np.all(
    np.abs(projected-positive_truth)<=np.abs(raw_values-positive_truth)+1e-12),
    "absolute and squared error cannot increase for nonnegative ground truth")

for tau in [.05,.10]:
    loge = rng.uniform(-.3,.3,10000)
    hit_raw = np.abs(np.exp(loge)-1)<=tau
    hit_log = (loge>=np.log(1-tau)) & (loge<=np.log(1+tau))
    check("relative_hit_equivalence_"+str(tau), np.array_equal(hit_raw,hit_log),
          {"lower_log_error":float(np.log(1-tau)),"upper_log_error":float(np.log(1+tau))})

steps=137
check("epoch_step_error_equivalence", np.allclose(
    np.abs(steps*prediction-steps*true_time)/(steps*true_time),relative),
    "same relative error when the epoch contract is exactly N times step mean")

# Memory units in this example are arbitrary, but identical throughout.
# Two views of storage A each occupy the same 100-unit backing storage.
storage = {"A":100.,"B":80.,"C":60.}
live = [("A","A","B"),("A","C")]
trace = [sum(storage[k] for k in set(items)) for items in live]
check("alias_aware_live_storage", trace==[180.,160.], trace)
reserved=np.array([100.,200.]); external=np.array([100.,0.])
check("max_of_sum_not_sum_of_max", (reserved+external).max()==200. and
      reserved.max()+external.max()==300.,
      {"correct_peak":200.,"sum_of_peaks":300.})

times=np.array([1.,3.])
best_constant=times.mean()
check("missing_state_irreducible_error", np.mean((times-best_constant)**2)==1.,
      {"same_observed_input_times":times.tolist(),"minimum_MSE":1.})

# A held-out direction never appears in calibration. Ridge cannot recover its effect.
phi_train=np.c_[np.ones(16),np.zeros(16)]
wu,_,_=ridge(phi_train,np.zeros(16),np.ones(16)/16,.001)
unseen=np.array([1.,1.]); true_unseen_log_residual=.5
check("unsupported_direction_counterexample", abs(unseen@wu)<1e-12 and
      abs(np.exp(unseen@wu-true_unseen_log_residual)-1)>.3,
      {"unseen_relative_error":float(abs(np.exp(unseen@wu-true_unseen_log_residual)-1))})

result={"passed":len(checks),"failed":0,"scope":"algebra and synthetic numerical checks only; no trained PerfSeer evaluation", "checks":checks}
print(json.dumps(result,indent=2))
