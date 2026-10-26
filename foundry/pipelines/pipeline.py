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
from mne_bids import get_bids_path_from_fname, read_raw_bids

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
        raws = []

        for path in download_output.paths:
            raw = read_raw_bids(
                get_bids_path_from_fname(path),
                extra_params={"preload": True},
            )

            # Match the larger pipeline: infer rest between annotated events.
            annotations = raw.annotations
            if "rest" not in annotations.description:
                for previous, following in zip(annotations[:-1], annotations[1:]):
                    start = previous["onset"] + previous["duration"]
                    end = following["onset"]

                    if end > start:
                        raw.annotations.append(start, end - start, "rest")

            raws.append(raw)

        raw = mne.concatenate_raws(raws)

        signal = IrregularTimeSeries(
            timestamps=raw.times,
            data=raw.get_data().T,
            domain="auto",
        )

        # This block sets labels of on vs off. What's important to note here
        # is that they're packaged in Interval.behavior_ids.
        starts, ends, labels = [], [], []

        for annotation in raw.annotations:
            description = annotation["description"].lower()

            if description == "rest" or "baseline" in description:
                label = 0
            elif "stim" in description and (
                "hz" in description
                or "white-noise" in description
                or "whitenoise" in description
            ):
                label = 1
            else:
                continue

            start = annotation["onset"] - raw.first_time
            end = start + annotation["duration"]

            if end <= start:
                continue

            # Divide long off/baseline periods into 0.5-second examples.
            boundaries = (
                np.append(np.arange(start, end, 0.5), end)
                if label == 0
                else np.array([start, end])
            )

            for left, right in zip(boundaries[:-1], boundaries[1:]):
                starts.append(left)
                ends.append(right)
                labels.append(label)

        if not labels:
            raise ValueError("No on/off trials found. Check the BIDS annotations.")

        starts = np.asarray(starts)
        ends = np.asarray(ends)
        labels = np.asarray(labels, dtype=np.int64)

        # Remove empty or rounding-sized trials BEFORE checking overlaps.
        keep = (ends - starts) > 1e-8
        starts, ends, labels = starts[keep], ends[keep], labels[keep]

        order = np.argsort(starts)
        starts, ends, labels = starts[order], ends[order], labels[order]

        overlap = ends[:-1] - starts[1:]

        if np.any(overlap > 1e-4):
            i = np.flatnonzero(overlap > 1e-4)[0]
            raise ValueError(
                f"Trials overlap by {overlap[i]:.6f}s: "
                f"({starts[i]}, {ends[i]}) and "
                f"({starts[i + 1]}, {ends[i + 1]})"
            )

        # Clip tiny floating-point overlaps.
        ends[:-1] = np.minimum(ends[:-1], starts[1:])

        # Remove any intervals made empty by clipping.
        keep = (ends - starts) > 1e-8
        starts, ends, labels = starts[keep], ends[keep], labels[keep]

        trials = Interval(
            start=starts,
            end=ends,
            timestamps=(starts + ends) / 2,
            behavior_ids=np.asarray(labels, dtype=np.int64),
            timekeys=["start", "end", "timestamps"],
        )
        trials.sort()

        channels = extract_channels(raw)

        # This saved structure is the interface used by the Dataset, tokenizer,
        # and task extractors. Its field names must agree with their configured paths.
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
            on_vs_off_trials=trials,
            domain=signal.domain,
        )

        # Save split intervals during preprocessing so subsequent training runs
        # can reuse the same partition without recreating it.
        data.train_domain, data.valid_domain, data.test_domain = trials.split(
            [0.7, 0.15, 0.15]
        )

        with h5py.File(output_path, "w") as f:
            data.to_hdf5(f)