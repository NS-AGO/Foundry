# Dataset and Module Implementation in Foundry

This guide walks through how to implement a model and a dataset in foundry. Both the model and the dataset pipeline are independent of each other and can be implemented without touching the other. The task config mediates the shared contact between model and data pipeline.

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

Some transforms need to calculate parameters from the dataset before they can be applied. In the example below. In these cases a `fit()` function is defined and called in `datamodule.setup()` before the final `Compose()` object is formed.

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

Foundry uses Hydra configuration files to define how datasets, ## DataModules, models, tasks, and training settings are assembled into an experiment. This allows the same Python implementation to be reused with different datasets, model architectures, hyperparameters, and tasks without modifying the source code.
When creating a new dataset, model, or experiment, the required configuration depends on which component is being added. In general, a new component should have a corresponding configuration file that specifies its class and the parameters needed to instantiate it.
dataset configuration specifies which Dataset class should be instantiated and provides the parameters needed to construct it.
For example:
\_target\_: foundry.datasets.MyDataset

recording_ids:
  - sub-01_ses-01
  - sub-02_ses-01

split_type: train
task_type: classification
The `_target_` field tells Hydra which Python class to instantiate. The remaining fields are passed to the class constructor.
Dataset configurations are useful for keeping dataset-specific information separate from the Dataset implementation. The Python class defines how the dataset works, while the configuration specifies which dataset and recordings should be used.
When creating a new Dataset class, a corresponding configuration should therefore normally be added under the appropriate dataset configuration directory.
The DataModule configuration specifies how the dataset is presented to the training pipeline.
Typical parameters include:
\_target\_: foundry.datamodules.MyDataModule

dataset:
  \_target\_: foundry.datasets.MyDataset
  recording_ids: [...]

batch_size: 32
num_workers: 8
sequence_length: 125
The DataModule configuration can also contain parameters controlling preprocessing, splitting, sampling, and transforms.
A model configuration specifies the model class and its architecture-specific hyperparameters. 
The task configuration defines what the model is being trained to predict and how its predictions are evaluated.
A task configuration typically specifies the target, loss, metrics, and readout-related settings. Conceptually:
name: frequency_classification

loss:
  \_target\_: foundry.losses.CrossEntropyLoss

metrics:
  - accuracy
  - f1
The exact structure depends on the task configuration system used by the experiment.
Task configuration is separate from the model because the same model architecture can be used for different tasks. For example, a neural encoder could produce the same representation while different task configurations define classification, regression, or other prediction objectives.
Training configuration specifies how the model is optimized rather than how the model itself is constructed.
Typical parameters include:

```yaml
learning_rate: 1e-4
weight_decay: 0.01

scheduler_name: phased
warmup: 1000
hold: 1000
decay: 10000
```
These parameters are consumed by the training module. The model itself should therefore generally not define its own optimizer or training loop.
This separation allows the same model to be trained with different optimization settings without changing the model implementation.
The experiment configuration brings the individual components together. 
data:
  datamodule:
    \_target\_: foundry.datamodules.MyDataModule

model:
  \_target\_: foundry.models.MyModel
  hidden_dim: 128

task:
  name: frequency_classification

training:
  learning_rate: 1e-4
  batch_size: 32
The experiment configuration therefore acts as the entry point for a particular training run. It determines which Dataset, DataModule, model, task, and training settings are used together.
Rather than creating a new Python training script for every experiment, different experiments can be created by composing different configuration files.


## Running on Foundry

To run on foundry you need to create a .toml file that points to a docker container. A baseline image has already been created in `a0091.ce-images` called `example-foundry.sqsh`. If you wish to create your own image you can follow the tutorial on the cscs documentation for [llm inference tutorial](https://docs.cscs.ch/tutorials/ml/llm-inference/#build-a-modified-ngc-pytorch-container).

To run a foundry session you can either spawn a srun and call

`uv run python main.py experiment=experiment_folder/experiment_file`

or run it through a sbatch command. An example sbatch script has been added to the repo as `run_example.sh`