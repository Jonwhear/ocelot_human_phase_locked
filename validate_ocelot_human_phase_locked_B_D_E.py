import numpy as np
import scipy.io
import hashlib

# 3/17/26 by Jon Whear (whear003@umn.edu AND jonwhear@gmail.com) with assistance by OpenAI ChatGPT 5.4

MAT_PATH = r"C:\Users\TNEL_Device_8\Downloads\widgePhaseDependentStimPhaseDependentData (2).mat"
NPZ_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_B_D_E_phase_locked_only.npz"
FS = 16000

SESSION_MATCH = {
    "B": ["instim03", "phasedependent", "session01"],
    "D": ["instim05", "phasedependent", "session01"],
    "E": ["instim05", "phasedependent", "session02"],
}

def normalize_name(x):
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")
    if isinstance(x, np.ndarray):
        if x.size == 1:
            return normalize_name(x.item())
        if x.dtype.kind in {"U", "S"}:
            return "".join(x.tolist()).strip()
    return str(x).strip()

def find_session_index(all_names, required_substrings):
    matches = []
    req = [s.lower() for s in required_substrings]
    for i, name in enumerate(all_names):
        low = name.lower()
        if all(s in low for s in req):
            matches.append(i)
    return matches

def as_1d(x, dtype=None):
    arr = np.atleast_1d(x).squeeze()
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr

def arr_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

def validate_event_indices(session_letter, stim_indices, trace_len):
    if len(stim_indices) == 0:
        raise ValueError(f"{session_letter}: no stimulation indices found")

    if np.any(stim_indices < 0):
        raise ValueError(f"{session_letter}: negative stimulation indices detected")

    if np.any(stim_indices >= trace_len):
        raise ValueError(f"{session_letter}: stimulation indices out of bounds for trace length {trace_len}")

    if not np.all(np.diff(stim_indices) >= 0):
        raise ValueError(f"{session_letter}: stimulation indices are not monotonically increasing")

    diffs = np.diff(stim_indices)
    if len(diffs) > 0:
        print(f"  inter-event spacing: min={diffs.min()}, mean={diffs.mean():.1f}, max={diffs.max()} samples")
    else:
        print("  only one stimulation event present")

mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)
allSessionNames = [normalize_name(x) for x in np.atleast_1d(mat["allSessionNames"])]
allPhaseTraces = np.atleast_1d(mat["allPhaseTraces"])
allStimTraces = np.atleast_1d(mat["allStimTraces"])
allStimTimeInd = np.atleast_1d(mat["allStimTimeInd"])

npz = np.load(NPZ_PATH, allow_pickle=False)

for session_letter, substrings in SESSION_MATCH.items():
    idx_matches = find_session_index(allSessionNames, substrings)
    if len(idx_matches) != 1:
        raise ValueError(f"{session_letter}: expected 1 match, found {idx_matches}")

    idx = idx_matches[0]

    src_phase = as_1d(allPhaseTraces[idx], dtype=np.float64)
    src_stim = as_1d(allStimTraces[idx], dtype=np.float64)
    src_stim_sample_indices = as_1d(allStimTimeInd[idx], dtype=np.int64)
    src_stim_times_sec = src_stim_sample_indices / FS

    out_phase = npz[f"{session_letter}_phase_trace"]
    out_stim = npz[f"{session_letter}_stim_trace"]
    if f"{session_letter}_stim_sample_indices" in npz.files:
        out_stim_sample_indices = npz[f"{session_letter}_stim_sample_indices"]
    elif f"{session_letter}_stim_times" in npz.files:
        out_stim_sample_indices = npz[f"{session_letter}_stim_times"]
    else:
        raise KeyError(
            f"No stimulation index array found for {session_letter}. "
            f"Available keys: {npz.files}"
        )
    if f"{session_letter}_stim_times_sec" in npz.files:
        out_stim_times_sec = npz[f"{session_letter}_stim_times_sec"]
    else:
        out_stim_times_sec = out_stim_sample_indices / FS

    print(f"\n{session_letter} -> {allSessionNames[idx]}")

    print("  phase exact:", np.array_equal(src_phase, out_phase))
    print("  stim exact:", np.array_equal(src_stim, out_stim))
    print("  stim sample indices exact:", np.array_equal(src_stim_sample_indices, out_stim_sample_indices))
    print("  stim times sec exact:", np.array_equal(src_stim_times_sec, out_stim_times_sec))

    print("  phase hash src/out:", arr_hash(src_phase), arr_hash(out_phase))
    print("  stim hash src/out:", arr_hash(src_stim), arr_hash(out_stim))
    print("  stim index hash src/out:", arr_hash(src_stim_sample_indices), arr_hash(out_stim_sample_indices))

    assert np.array_equal(src_phase, out_phase), f"{session_letter} phase mismatch"
    assert np.array_equal(src_stim, out_stim), f"{session_letter} stim mismatch"
    assert np.array_equal(src_stim_sample_indices, out_stim_sample_indices), f"{session_letter} stim sample indices mismatch"
    assert np.array_equal(src_stim_times_sec, out_stim_times_sec), f"{session_letter} stim times sec mismatch"

    validate_event_indices(session_letter, out_stim_sample_indices, len(out_phase))

    print(f"  first 10 stim sample indices: {out_stim_sample_indices[:10]}")
    print(f"  first 10 stim times sec: {out_stim_times_sec[:10]}")

print("\nValidation passed: exported phase traces, stim traces, and stimulation indices match the source MATLAB file exactly.")
