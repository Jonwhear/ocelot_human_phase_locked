import numpy as np
import scipy.io
import json
import os

# 3/17/26 by Jon Whear (whear003@umn.edu AND jonwhear@gmail.com) with formatting assistance by OpenAI ChatGPT 5.4

# =========================
# SETTINGS
# =========================

MAT_PATH = r"C:\Users\TNEL_Device_8\Downloads\widgePhaseDependentStimPhaseDependentData (2).mat"
OUT_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_B_D_E_phase_locked_only.npz"
MANIFEST_PATH = r"C:\Users\TNEL_Device_8\Downloads\ocelot_B_D_E_phase_locked_only_manifest.json"
FS = 16000

# Selected Human PL stimulation sessions
# B = instim03 PL
# D = instim05 session01 PL
# E = instim05 session02 PL

# Adjust these substrings if needed after printing allSessionNames.
SESSION_MATCH = {
    "B": ["instim03", "phaseDependent", "session01"],
    "D": ["instim05", "phaseDependent", "session01"],
    "E": ["instim05", "phaseDependent", "session02"],
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


# =========================
# FUNCTIONS
# =========================

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


# =========================
# LOAD FILE
# =========================

mat = scipy.io.loadmat(MAT_PATH, squeeze_me=True, struct_as_record=False)

allSessionNames = [normalize_name(x) for x in np.atleast_1d(mat["allSessionNames"])]
allPhaseTraces = np.atleast_1d(mat["allPhaseTraces"])
allStimTraces = np.atleast_1d(mat["allStimTraces"])
allStimTimeInd = np.atleast_1d(mat["allStimTimeInd"])

print("\nAvailable compiled sessions:")
for i, name in enumerate(allSessionNames):
    print(f"[{i}] {name}")

# =========================
# EXTRACT B, D, E
# =========================

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
        raise ValueError(
            f"No compiled session matched {session_letter} with substrings {substrings}.\n"
            f"Check printed session names and tighten the matching."
        )
    if len(idx_matches) > 1:
        raise ValueError(
            f"Multiple compiled sessions matched {session_letter}: {idx_matches}\n"
            f"Matched names: {[allSessionNames[i] for i in idx_matches]}\n"
            f"Tighten SESSION_MATCH so each letter maps to exactly one PL session."
        )

    idx = idx_matches[0]
    compiled_name = allSessionNames[idx]

    phase_trace = as_1d(allPhaseTraces[idx], dtype=np.float64)
    stim_trace = as_1d(allStimTraces[idx], dtype=np.float64)
    stim_times = as_1d(allStimTimeInd[idx], dtype=np.int64)

    if phase_trace.shape != stim_trace.shape:
        raise ValueError(
            f"{session_letter}: phase/stim trace length mismatch: "
            f"{phase_trace.shape} vs {stim_trace.shape}"
        )

    # Save arrays into the NPZ under simple names
    export_dict[f"{session_letter}_phase_trace"] = phase_trace
    export_dict[f"{session_letter}_stim_trace"] = stim_trace
    export_dict[f"{session_letter}_stim_times"] = stim_times

    manifest["sessions"][session_letter] = {
        "compiled_session_name": compiled_name,
        "n_samples": int(len(phase_trace)),
        "n_stim_events": int(len(stim_times)),
        **SESSION_METADATA[session_letter],
    }

    print(f"\n{session_letter}:")
    print(f"  compiled session = {compiled_name}")
    print(f"  samples = {len(phase_trace)}")
    print(f"  stim events = {len(stim_times)}")

# =========================
# SAVE OUTPUTS
# =========================

np.savez_compressed(OUT_PATH, **export_dict)

with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"\nSaved NumPy archive to:\n{OUT_PATH}")
print(f"Saved manifest to:\n{MANIFEST_PATH}")
