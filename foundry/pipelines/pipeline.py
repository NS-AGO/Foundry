from argparse import ArgumentParser

import h5py
import mne
import numpy as np
import pandas as pd

from torch_brain.data import (
    Data,
    Interval,
    IrregularTimeSeries,
    BrainsetDescription,
    SessionDescription,
    SubjectDescription,
    DeviceDescription,
)
from torch_brain.pipeline import BrainsetPipeline

from torch_brain.utils.mne import extract_channels


parser = ArgumentParser()
parser.add_argument("--reprocess", action="store_true")


class Pipeline(BrainsetPipeline):
    brainset_id = "neurosoft_monkeys_2026"
    parser = parser

    @classmethod
    def get_manifest(cls, raw_dir, args):
        # One row per session
        sessions = {}
        for path in sorted(raw_dir.rglob("*_ieeg.vhdr")):
            parts = path.name.removesuffix("_ieeg.vhdr").split("_")
            keep = [p for p in parts if p.split("-")[0] in ("sub", "ses", "task", "desc")]
            sessions.setdefault("_".join(keep), []).append(str(path))

        if not sessions:
            raise ValueError(f"No iEEG recordings found in {raw_dir}")

        return pd.DataFrame(
            {"session_id": list(sessions), "paths": list(sessions.values())}
        ).set_index("session_id")

    def download(self, manifest_item):
        # The data is already local, so there is nothing to download.
        return manifest_item

    def process(self, download_output):
        session_id = download_output.Index
        output_path = self.processed_dir / f"{session_id}.h5"

        if output_path.exists() and not self.args.reprocess:
            return

        # Stitch all recordings of the session together in order.
        raws = [
            mne.io.read_raw_brainvision(p, preload=True)
            for p in download_output.paths
        ]
        raw = mne.concatenate_raws(raws)

        signal = IrregularTimeSeries(
            timestamps=raw.times,
            data=raw.get_data().T,
            domain="auto",
        )

        # Each "Stimulus/..." marker is a stimulus onset (BrainVision markers
        # have ~zero duration, so a trial runs from one onset to the next).
        onsets = np.array(
            [a["onset"] for a in raw.annotations if a["description"].startswith("Stimulus")]
        )
        if len(onsets) < 4:
            found = sorted(set(raw.annotations.description))
            raise ValueError(
                f"{session_id}: found {len(onsets)} 'Stimulus' annotations. "
                f"Annotation descriptions in this file: {found}"
            )
        ends = np.append(onsets[1:], raw.times[-1])
        trials = Interval(start=onsets, end=ends)

        channels = extract_channels(raw)

        data = Data(
            brainset=BrainsetDescription(
                id=self.brainset_id,
                origin_version="0.0.1",
                derived_version="1.0.0",
                source="BIDS",
                description="Simple ECoG dataset for a linear model example.",
            ),
            subject=SubjectDescription(
                id=session_id.split("_")[0],  # e.g. "sub-01"
                species="unknown",
            ),
            session=SessionDescription(id=session_id),
            device=DeviceDescription(id=session_id),
            ecog=signal,
            channels=channels,
            trials=trials,
            domain=signal.domain,
        )

        # Chronological 70 / 15 / 15 split of the trials.
        data.train_domain, data.valid_domain, data.test_domain = trials.split(
            [0.7, 0.15, 0.15]
        )

        with h5py.File(output_path, "w") as f:
            data.to_hdf5(f)