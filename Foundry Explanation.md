# Dataset and Module Implementation in Foundry

This guide explains how to implement a dataset and a model in Foundry. The preprocessing pipeline and model can be developed separately, provided they agree on the data interface. The pipeline defines the saved signal and label fields, the tokenizer converts sampled data into model inputs and targets, and the task configuration defines how targets are extracted, predicted, and evaluated.

> **Figure:** Add an overview image here.

## Brainset Pipeline

> **Figure:** Add the dataset pipeline overview image here.

The dataset pipeline for foundry is a run once, offline step. Foundry never works with raw BIDS/NWB/EDF files directly, instead these are converted into HDF5 files. This is completed through torch brain by implementing `torch_brian.pipeline`. This pipeline describes what data should be processed, how individual recordings are obtained, and how those recordings are transformed into `Data` objects.

A brainset pipeline is created by implementing the abstract `BrainsetPipeline` class. It follows the basic structure:

```python
from torch_brain.pipeline import BrainsetPipeline

class MyPipeline(BrainsetPipeline):
    brainset_id = "my_brainset"

    @classmethod
    def get_manifest(cls, raw_dir, args):
        ...

    def download(self, manifest_item):
        ...

    def process(self, download_output):
        ...
```

Detailed information about each function can be found in the [TorchBrain BrainsetPipeline documentation](https://torch-brain.readthedocs.io/en/latest/generated/api/autosummary/torch_brain.pipeline.BrainsetPipeline.html). 

As a general overview, `get_manifest()` creates a manifest in the form of a pandas DataFrame. The manifest defines the individual processing units for the pipeline and contains the information needed to identify the raw data associated with each unit. Each row of the DataFrame is a `manifest_item` which is passed to `download()` and `process()`. In other words it tells the pipeline what data belongs together. Each row is one unit.

For example in the Neurosoft pipeline, the manifest groups recordings that belong to the same session and associates each `session_id` with the corresponding `recordings_ids`.

The `download()` function verifies that all the data listed in the `manifest_item` are present in the raw data directory. It doesn’t actually download any data, just verifies a match between the manifest and a file exists.

The `process()` function is where the data is transformed into a `Data` object. Further information on constructing a data object can be found in the [brainset documentation](https://torch-brain.readthedocs.io/en/latest/generated/api/autosummary/torch_brain.data.Data.html). The end of `process()` should save the `Data` object to an HDF5 file which will then be loaded by foundry.

A `Data` object is a standardized container for a dataset, but its contents can be adapted to the needs of the particular dataset. Standard information such as subjects, sessions, channels, and neural signals can be combined with custom fields. The core objects for a `Data` object are the `Interval`, `RegularTimeSeries`, and `IrregularTimeSeries`. Interval objects describe specific time periods within a recording, such as trials or stimulus presentations. They typically contain the start and end times of each interval, along with additional information describing that event. Regular time series represent data sampled at regular intervals, meaning the time between consecutive measurements is constant. Irregular time series represent data where measurements occur at non-uniform time intervals. Instead of assuming a constant sampling rate, each measurement is associated with its own timestamp.

Finally, a TorchBrain dataset is created through CLI commands. You first need to configure the pipeline by calling `brainsets config`. Then to run the pipeline `brainsets prepare` is called. Make sure to have the raw directory point to the parent folder of the raw dataset and to have `brainset_id` in your pipeline match the name of your dataset folder. Further details on the CLI commands can be found in the [documentation](https://torch-brain.readthedocs.io/en/latest/cli/commands.html).

## Datasets

The dataset is the data access step that Foundry uses to access the processed data produced by the dataset pipeline. Rather than working directly with the original BIDS/NWB/EDF files, the Dataset reads the processed HDF5 files containing `Data` objects. It provides an interface for accessing complete recordings or specific time windows from those recordings. The `Dataset` class is also the layer to implement dataset specific functions such as how you wish the data to split itself.
A TorchBrain dataset is created by using or subclassing the `Dataset` class:

```python
from torch_brain.datasets import Dataset

class MyDataset(Dataset):
    ...
```

Detailed information about the `Dataset` class can be found in the [TorchBrain Dataset documentation](https://torch-brain.readthedocs.io/en/stable/generated/api/autosummary/torch_brain.dataset.Dataset.html).
Because a `dataset` object is reading a `Data` object the fields of a dataset should correspond to the fields established in the TorchBrain pipeline. The Dataset is given a directory containing the processed HDF5 recording files. Each HDF5 file corresponds to a recording and is identified by its `recording_id`. The Dataset can either load all recordings in the directory or a specified list of `recording_ids`. 
A complete recording can be accessed using:

```python
data = dataset.get_recording(recording_id)
```

This returns the corresponding `Data` object. Importantly, the data is lazy-loaded, meaning the entire recording does not need to be loaded into memory immediately. Only the attributes or time ranges that are actually requested are loaded.
The Dataset itself does not decide what the individual training samples are. Instead, it provides access to recordings and defines which time intervals are available for sampling.

`get_sampling_intervals()` returns the time intervals that can be sampled from each recording. By default, these intervals cover the entire recording, but a custom Dataset can override this behaviour to restrict sampling to particular intervals, such as trials.

Samplers then use these intervals to determine the individual samples used during training. For example, a sampler can select fixed-length windows from the available intervals. In other words the dataset defines what data is available and the sampler selects what becomes a sample.

A `DatasetIndex` specifies a particular recording and a start and end time. It tells the Dataset exactly which portion of a recording should be loaded. The Dataset uses this information to return a `Data` object containing that  section of the recording.
This is different from a standard PyTorch Dataset, where the Dataset itself generally defines the individual samples. In TorchBrain, the Dataset provides access to the recordings, while samplers determine how those recordings are divided into samples.
A Dataset can also be subclassed when a dataset requires behaviour beyond simply loading the HDF5 recordings. For example, `get_sampling_intervals()` can be overridden to specify custom sampling regions, and `get_recording_hook()` can be overridden to perform lightweight processing immediately after a recording is loaded.

This allows a Dataset to contain specific logic while keeping the underlying `Data` objects standardized.

Field names are part of the interface between preprocessing and training. They are configurable, but every component must use matching paths. The working example uses:

| Saved field | Purpose |
|---|---|
| `ecog.data` | Neural signal with shape `(time, channels)` |
| `channels` | Channel identifiers and metadata |
| `on_vs_off_trials.timestamps` | Target times, stored at interval midpoints |
| `on_vs_off_trials.behavior_ids` | Integer targets: off = 0, on = 1 |
| `train_domain`, `valid_domain`, `test_domain` | Intervals available for each split |

If preprocessing changes a field name, update the tokenizer or task extractor accordingly and regenerate the processed files. Changing Python code does not modify existing HDF5 files.

## DataModules

The DataModule provides the interface between the Dataset and the model training pipeline. While the Dataset determines which recordings and time intervals are available for sampling, the DataModule configures how those samples are prepared and delivered during training.
A DataModule typically specifies the Dataset class, batch size, number of workers, sequence length, and any transforms applied to the neural data. The DataModule selects which Dataset class to use and passes down arguments like the `recording_ids`, `split_type`, `task_type`, and `fold_number`. 
Preprocessing is performed in Foundry using transforms. Transforms are called sequentially in the `Compose([transform 1, transform 2, ...])` function by setting:

```python
self.dataset.transforms = Compose([transform 1, transform 2, ...]) 
```

Transforms have three main components:

- an `__init__()` method for configuration.
- an optional `fit()` method for learning parameters from the dataset.
- a `__call__()` method that applies the transformation to a `Data` object.

Some transforms learn parameters from data before they can be applied. For example, a standardization transform may estimate a mean and standard deviation. Fit these parameters using only samples inside the training intervals, then reuse them for validation and testing. Reading entire recordings during fitting can include validation or test data, even when `split="train"` was requested.

The DataModule should attach each transform only once, because `setup()` may be called more than once during a run. The provided `ExampleTransform` is a placeholder: it currently returns its input unchanged.

## Tokenization and Batching

In Foundry, tokenization means converting a sampled `Data` object into the dictionary consumed by the training pipeline. It does not necessarily involve dividing signals into patches or assigning vocabulary IDs. A simple MLP still needs this preparation step to extract its input features and labels.

In the example, `tokenize()` reads `data.ecog.data`, which has shape `(T, C)`, and averages over time to produce one value per channel. Batching then combines these vectors into a tensor of shape `(B, C)`. The MLP therefore uses the mean signal within each window and discards its temporal structure.

The tokenizer also returns `task_index`, `target_values`, `target_weights`, and metadata such as `session_id` and `absolute_start`. `extract_multitask_targets()` prepares targets using the configured target extractors. The batching wrappers `pad8()` and `chain()` allow samples with different numbers of targets to be combined.

Task indices are one-based, with zero reserved for padding. The example model removes padded entries and subtracts one before passing task indices to the readout router. Transforms that operate on Data objects must run before tokenization, because tokenization returns a dictionary.

## Model Creation

Models are implemented in `foundry.models` and define how neural data is transformed into task predictions. Foundry models are standard PyTorch `nn.Module` classes; the model is responsible for the architecture and forward pass, while the training logic is handled separately by `FoundryModule`.
A model typically consists of:

- an `__init__()` method that defines the model architecture and its configurable hyperparameters;
- a `forward()` method that receives the tensors produced by the DataModule and computes predictions;
- optionally, helper methods such as `extract_features()` for exposing intermediate representations.

The DataModule produces batches as dictionaries. Foundry removes the target and metadata fields before calling the model, and passes the remaining fields directly to `forward()`.
For example, a model may receive:

```python
def forward(
    self,
    *,
    input_values: torch.Tensor,
    task_index: torch.Tensor,
    **kwargs,
):
    ...
```
  
`input_values` contains the neural signal, while `task_index` identifies which task or readout should be applied to each output. Additional fields can be passed through `**kwargs`, for example information required by a channel strategy.
The model should return predictions in the format expected by the task readout. In the standard Foundry models this is a dictionary mapping task names to prediction tensors:

```
return self._route_readout(x, task_index)
```

The exact output dimensions depend on the task configuration. For a classification task with `N` classes, for example, the model produces `N` logits for each prediction.
Foundry models follow the structure of having an `__init__()` method for configuration and creation of the model architecture and shape, and a `forward()` function for computation. Models can be built from standard PyTorch modules such as `nn.Conv1d`, `nn.Conv2d`, `nn.Linear`, `nn.GRU`, `nn.Dropout`, and `nn.BatchNorm`. 
Models should explicitly handle the shape of the input they expect. Foundry's baseline models generally normalize neural data to `(B, C, T)`, where:

- `B` = batch size
- `C` = number of channels
- `T` = number of time samples

When implementing a new model, the expected input shape should therefore be clearly defined and any necessary conversion should happen inside the model rather than being in the DataModule. 
Foundry separates the neural representation produced by the model from the final task-specific prediction. Models generally produce an embedding and pass it through a `ReadoutRouter`. The router uses the task configuration to select the appropriate readout head. This allows the same backbone to support different tasks without hard-coding the number of classes or output dimensions into the architecture.
Consequently, a model should generally not implement a task-specific loss or classifier training procedure itself. It should produce the predictions required by the configured task.
Loss functions are defined by the task configuration rather than inside the model. During initialization, `FoundryModule` reads the loss specified for each task and instantiates it:
self.\_task_losses[name] = instantiate(cfg.loss)
During each training, validation, or test step, `FoundryModule`:

1. unpacks the batch;
2. calls the model's `forward()` method;
3. obtains the model predictions;
4. computes the configured loss for each task;
5. combines the task losses;
6. updates the configured metrics.

This means that model implementations should not normally contain a `backward()` or `backwards()` function. Backpropagation is handled by PyTorch Lightning after the training step returns the loss. The model's responsibility is to define the differentiable computation that produces predictions. Rather the training module determines how those predictions are evaluated and optimized.
The same model can therefore be used with different losses or task configurations without modifying the model implementation.
The optimizer is also configured outside the model. `FoundryModule.configure_optimizers()` creates the `AdamW` optimizer and optionally configures learning-rate schedules such as warmup, hold, or cosine decay.
For this reason, a new model generally should not define its own optimizer or training loop. Hyperparameters such as learning rate, weight decay, and scheduler settings belong in the training configuration.
## Configuration Files

Foundry uses Hydra configuration files to select and configure the DataModule, model, tasks, and training settings. An experiment config brings these components together, allowing different experiments to use the same Python implementations.

The examples below follow the working example implementation.

### DataModule Configuration

The DataModule config specifies which DataModule and Dataset classes to use, where the processed data is stored, and how samples are loaded.

The file `configs/data/example/example_dataconfig.yaml` contains:

```yaml
_target_: foundry.data.datamodules.ExampleDataModule

dataset_class: foundry.data.datasets.ExampleDataset

root: ./data/processed
batch_size: ${hyperparameters.batch_size}
num_workers: ${hyperparameters.num_workers}
sequence_length: ${hyperparameters.sequence_length}

recording_ids:
  - sub-01_ses-01_task-AcousticStim_desc-raw
```
`_target_` identifies the Python class that Hydra instantiates. The remaining entries configure that class.

`root` points to the parent directory containing the processed dataset. In this example, `ExampleDataset` appends `neurosoft_monkeys_2026`, so the recording files are read from:

```
./data/processed/neurosoft_monkeys_2026/
```

`recording_ids` selects the recordings to load. Each identifier corresponds to an HDF5 filename without its `.h5` extension.

`sequence_length` is the sampling-window duration in seconds. A value of `0.5` requests half-second windows.

Values such as `${hyperparameters.batch_size}` are Hydra interpolations: their values are taken from the corresponding fields in the experiment configuration.

### Model Configuration
The model config selects the model class and its architecture settings.

The file `configs/model/example_model.yaml` contains:

```yaml
_target_: foundry.models.ExampleMLP

num_channels: 32
hidden_dim: 128
```

Both, `num_channels` and `hidden_dim` match the arguments required when constructing the `ExampleMLP` model. The model yaml should mainy aim to fill out parameters for the `__init__()` class for your model. Foundry already handles the `task_config` handover, so this should not be included in the model config.

### Task Configuration
The task config defines what the model predicts, how labels are extracted, and how predictions are evaluated.

The file `configs/tasks/example_task.yaml` contains:

```yaml
name: example_task

head:
  _target_: foundry.tasks.heads.ReadoutHead
  output_dim: 2

target_extractor:
  _target_: foundry.tasks.targets.TargetExtractor
  timestamp_key: on_vs_off_trials.timestamps
  value_key: on_vs_off_trials.behavior_ids

metrics:
  _target_: foundry.tasks.example_metrics.example_classification_metrics
  num_classes: 2

loss:
  _target_: foundry.tasks.losses.CrossEntropyTaskLoss

metric_summary_modes:
  acc: max
  loss: min
```

`head` defines the task-specific readout. This example uses two output logits because the task distinguishes stimulation off from stimulation on.

`target_extractor` specifies the paths to the target timestamps and values inside each sampled Data object. These paths must match the fields saved in the constuction of the data object by your `pipeline.py`. In this example, behavior_ids contains integer labels: off = 0 and on = 1.

`metrics` defines the evaluation metrics, while `loss` defines the training objective. The metric class count must agree with the readout output dimension.

`metric_summary_modes` specifies whether larger or smaller values are preferred when summarizing metrics in W&B. It does not configure early stopping; that is controlled separately by the trainer callbacks.

## Experiment Configuration
The experiment config selects the component configurations and provides settings for a particular run. It acts as the main integrating config file for a Foundry run.

The file `configs/experiment/example/example_experiment.yaml` contains:

```yaml
# @package _global_

defaults:
  - override /model: example_model
  - override /data: example/example_dataconfig

task_configs:
  - example_task

model:
  num_channels: ${hyperparameters.num_channels}

run:
  project: example_project
  name: example_experiment
  group: example
  seed: 42
  tags:
    - examples

hyperparameters:
  batch_size: 16
  sequence_length: 0.5
  num_workers: 4
  learning_rate: 0.001
  weight_decay: 0.0
  num_channels: 32

trainer:
  max_epochs: 100
  precision: bf16-mixed
  callbacks:
    early_stopping:
      monitor: val/example_task_acc
      mode: max
      patience: 10

stage:
  mode: direct
```

The `defaults` list selects the model and `DataModule` configs. Paths are relative to their configuration groups: `example_model` selects `configs/model/example_model.yaml`, while `example/example_dataconfig` selects `configs/data/example/example_dataconfig.yaml`.

`task_configs` selects the task files to load. Foundry uses their definitions to construct the task readouts, target extractors, losses, and metrics.

`hyperparameters` supplies values referenced elsewhere in the configuration.

`trainer` overrides the default PyTorch Lightning trainer settings. Here, early stopping monitors validation accuracy and stops after ten validation checks without sufficient improvement. The monitored metric name must match a metric produced by the selected task.

`stage.mode: direct` uses the configured data location directly instead of staging a copy.

Run the experiment from the repository root with:

```Bash
uv run python main.py experiment=example/example_experiment
```

Individual settings can also be overridden from the command line:

```Bash
uv run python main.py \
  experiment=example/example_experiment \
  hyperparameters.batch_size=32 \
  trainer.max_epochs=20
```

This keeps experiment choices in configuration files while leaving the dataset, model, and training implementations reusable.

## Running on Foundry

To run on foundry you need to create a .toml file that points to a docker container. A baseline image has already been created in `a0091.ce-images` called `example-foundry.sqsh`. If you wish to create your own image you can follow the tutorial on the cscs documentation for [llm inference tutorial](https://docs.cscs.ch/tutorials/ml/llm-inference/#build-a-modified-ngc-pytorch-container).

To run a foundry session you can either spawn a srun and call

```Bash
uv run python main.py experiment=example/example_experiment
```

or run it through a sbatch command. An example sbatch script has been added to the repo as `run_example.sh`