import numpy as np
import scipy.io
import hashlib

# 3/17/26 by Jon Whear (whear003@umn.edu AND jonwhear@gmail.com) with assistance by OpenAI ChatGPT 5.4

''' VALIDATE SUMMARY
- load the original compiled MATLAB file and the exported NumPy .npz file
- identify the phase-locked sessions corresponding to B, D, and E in the compiled MATLAB file
- extract the original phase trace, stimulation trace, and stimulation times for each session
- load the exported phase trace, stimulation trace, and stimulation times for B, D, and E from the .npz
- compare source vs exported arrays for exact equality in shape and values
- optionally compute hashes or print summary stats as an extra integrity check
- raise an error if any session does not match exactly; otherwise report validation passed
'''

MAT_PATH = r"C:\Users\TNEL_Device_8\Downloads\widgePhaseDependentStimPhaseDependentData (2).mat"
NPZ_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_B_D_E_phase_locked_only.npz"

SESSION_MATCH = {
    "B": ["instim03", "phaseDependent", "session01"],
    "D": ["instim05", "phaseDependent", "session01"],
    "E": ["instim05", "phaseDependent", "session02"],
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
        if all(s.lower() in low for s in req):
            matches.append(i)
    return matches

def as_1d(x, dtype=None):
    arr = np.atleast_1d(x).squeeze()
    if dtype is not None:
        arr = np.asarray(arr, dtype=dtype)
    return arr

def arr_hash(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()

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
    src_times = as_1d(allStimTimeInd[idx], dtype=np.int64)

    out_phase = npz[f"{session_letter}_phase_trace"]
    out_stim = npz[f"{session_letter}_stim_trace"]
    out_times = npz[f"{session_letter}_stim_times"]

    print(f"\n{session_letter} -> {allSessionNames[idx]}")
    print("phase exact:", np.array_equal(src_phase, out_phase))
    print("stim  exact:", np.array_equal(src_stim, out_stim))
    print("times exact:", np.array_equal(src_times, out_times))

    print("phase hash src/out:", arr_hash(src_phase), arr_hash(out_phase))
    print("stim  hash src/out:", arr_hash(src_stim), arr_hash(out_stim))
    print("times hash src/out:", arr_hash(src_times), arr_hash(out_times))

    assert np.array_equal(src_phase, out_phase), f"{session_letter} phase mismatch"
    assert np.array_equal(src_stim, out_stim), f"{session_letter} stim mismatch"
    assert np.array_equal(src_times, out_times), f"{session_letter} stim_times mismatch"

print("\nValidation passed: exported arrays are identical to the source MATLAB arrays.")
