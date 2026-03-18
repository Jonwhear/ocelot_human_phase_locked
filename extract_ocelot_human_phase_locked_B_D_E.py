import numpy as np
import scipy.io
import json
import os

# 3/17/26 by Jon Whear (whear003@umn.edu AND jonwhear@gmail.com) with formatting assistance by OpenAI ChatGPT 5.4

MAT_PATH = r"C:\Users\TNEL_Device_8\Downloads\widgePhaseDependentStimPhaseDependentData (2).mat"
OUT_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_human_phase_locked_validation_set_B_D_E.npz"
MANIFEST_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_human_phase_locked_validation_set_B_D_E_manifest.json"
FS = 16000

SESSION_MATCH = {
    "B": ["instim03", "phasedependent", "session01"],
    "D": ["instim05", "phasedependent", "session01"],
    "E": ["instim05", "phasedependent", "session02"],
}

SESSION_METADATA = {
    "B": {
        "phase_structure": "Left Temporal Cortex (BA21)",
        "stim_structure": "Left Amygdala",
        "phase_channel_labels": ["LPH9", "LPH10"],
        "stim_channel_labels": ["LA2", "LA3"],
    },
    "D": {
        "phase_structure": "Right OFC (BA11)",
        "stim_structure": "Right Amygdala",
        "phase_channel_labels": ["ROF3", "ROF4"],
        "stim_channel_labels": ["RAM2", "RAM3"],
    },
    "E": {
        "phase_structure": "Left Temporal Pole (BA21)",
        "stim_structure": "Left Temporal Pole (BA38)",
        "phase_channel_labels": ["LA6", "LA7"],
        "stim_channel_labels": ["LPH10", "LPH11"],
    },
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

mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)

allSessionNames = [normalize_name(x) for x in np.atleast_1d(mat["allSessionNames"])]
allPhaseTraces = np.atleast_1d(mat["allPhaseTraces"])
allStimTraces = np.atleast_1d(mat["allStimTraces"])
allStimTimeInd = np.atleast_1d(mat["allStimTimeInd"])

print("\nAvailable compiled sessions:")
for i, name in enumerate(allSessionNames):
    print(f"[{i}] {name}")

export_dict = {}
manifest = {
    "source_matlab_file": MAT_PATH,
    "sampling_rate_hz": FS,
    "description": "Phase-locked human stimulation sessions B, D, and E extracted from compiled MATLAB container.",
    "sessions": {}
}

for session_letter, substrings in SESSION_MATCH.items():
    idx_matches = find_session_index(allSessionNames, substrings)

    if len(idx_matches) == 0:
        raise ValueError(f"No compiled session matched {session_letter} with substrings {substrings}")
    if len(idx_matches) > 1:
        raise ValueError(
            f"Multiple compiled sessions matched {session_letter}: {idx_matches}\n"
            f"Matched names: {[allSessionNames[i] for i in idx_matches]}"
        )

    idx = idx_matches[0]
    compiled_name = allSessionNames[idx]

    phase_trace = as_1d(allPhaseTraces[idx], dtype=np.float64)
    stim_trace = as_1d(allStimTraces[idx], dtype=np.float64)
    stim_sample_indices = as_1d(allStimTimeInd[idx], dtype=np.int64)
    stim_times_sec = stim_sample_indices / FS

    if phase_trace.shape != stim_trace.shape:
        raise ValueError(
            f"{session_letter}: phase/stim trace length mismatch: "
            f"{phase_trace.shape} vs {stim_trace.shape}"
        )

    export_dict[f"{session_letter}_phase_trace"] = phase_trace
    export_dict[f"{session_letter}_stim_trace"] = stim_trace
    export_dict[f"{session_letter}_stim_sample_indices"] = stim_sample_indices
    export_dict[f"{session_letter}_stim_times_sec"] = stim_times_sec

    manifest["sessions"][session_letter] = {
        "compiled_session_name": compiled_name,
        "n_samples": int(len(phase_trace)),
        "n_stim_events": int(len(stim_sample_indices)),
        "first_10_stim_sample_indices": stim_sample_indices[:10].tolist(),
        "first_10_stim_times_sec": stim_times_sec[:10].tolist(),
        **SESSION_METADATA[session_letter],
    }

    print(f"\n{session_letter}:")
    print(f"  compiled session = {compiled_name}")
    print(f"  samples = {len(phase_trace)}")
    print(f"  stim events = {len(stim_sample_indices)}")

np.savez_compressed(OUT_PATH, **export_dict)

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\nSaved NumPy archive to:\n{OUT_PATH}")
print(f"Saved manifest to:\n{MANIFEST_PATH}")
