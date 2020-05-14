## PolyAMP: Poly-Instrumental Audio to Midi Prediction
General multi-instrument audio transcription models.



When running, it's helpful to specify `--model_id` so that checkpoints for
that id are saved in one place. If a saved id is set with `--load_id`,
the latest checkpoint from training that model will be loaded.
Otherwise, `--load_id` defaults to "`*`", and whatever is the most recent
checkpoint will be loaded.

Additionally, hyperparameters can be overridden by specifying json for `--hparams`.
For example:\
`--hparams={\"frames_true_weighing\":3,\"frame_prediction_threshold\":0.4}`

Common hyperparameter overrides include:
- `predict_frame_threshold`
- `predict_onset_threshold`
- `predict_offset_threshold`
- `active_onset_threshold`
- `frames_true_weighing`
- `onsets_true_weighing`
- `offsets_true_weighing`
- `multiple_instruments_threshold`
- `use_all_instruments`
- `family_recall_weight`

`--load_full` can be specified to load the weights from Full Model training if _True_ or
load the weights from the individual Melodic and/or Timbre Models if _False_.

## Training
To train PolyAMP yourself, you can individually train the Melodic and Timbre Models.
Their weights can later be loaded into the Full Model (using `--load_full=False`).
The Full Model can also be trained from scratch.
### Melodic Model
For training on the Maestro dataset:
```bash
python polyamp_runner.py \
    --examples_path=/path/to/maestro/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=MELODIC \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD}
```
### Timbre Model
For training on the NSynth dataset:
```bash
python polyamp_runner.py \
    --examples_path=/path/to/nsynth/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=TIMBRE \
    --dataset_type=nsynth \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD}
```
For training on the Slakh dataset:
```bash
python polyamp_runner.py \
    --examples_path=/path/to/slakh/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=TIMBRE \
    --dataset_type=slakh \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD}
```
### Full Model
For training on the Maestro, Slakh, Hand-Curated, and NSynth datasets:
```bash
python polyamp_runner.py \
    --examples_path=/path/to/maestro/train.tfrecord-*,/slakh/train.tfrecord-*,/custom/train.tfrecord-* \
    --nsynth_examples_path=/path/to/nsynth/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=FULL \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD}
```

## Evaluating
Evaluating the Melodic Model provides metrics for the instrument-agnostic predictions.

Evaluating the Timbre Model provides metrics for instrument labelling.

Evaluating the Full Model provides metrics for the combined multi-track transcription task.
### Melodic Model
To evaluate the Melodic Model on the Maestro dataset:
```bash
python polyamp_runner.py \
    --mode=eval \
    --examples_path=/path/to/maestro/test.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=MELODIC \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD} \
    --eval_num_steps={NUMBER_OF_SAMPLES_TO_EVALUATE}
```
To evaluate the Full Model's agnostic predictions on the Maestro dataset:
```bash
python polyamp_runner.py \
    --mode=eval \
    --examples_path=/path/to/maestro/test.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=MELODIC \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD} \
    --eval_num_steps={NUMBER_OF_SAMPLES_TO_EVALUATE} \
    --load_full=True # Loads weights from Full Model
```
### Timbre Model
To evaluate the Timbre Model on the Slakh dataset:
```bash
python polyamp_runner.py \
    --mode=eval \
    --examples_path=/path/to/slakh/test.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=TIMBRE \
    --model_id=ANY_ID_STRING \
    --dataset_type=slakh \
    --load_id={ID_TO_LOAD} \
    --eval_num_steps={NUMBER_OF_SAMPLES_TO_EVALUATE}
```
### Full Model
To evaluate the Full Model on the Slakh dataset:
```bash
python polyamp_runner.py \
    --mode=eval
    --examples_path=/path/to/slakh/test.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=FULL \
    --model_id={ANY_ID_STRING} \
    --load_id={ID_TO_LOAD} \
    --eval_num_steps={NUMBER_OF_SAMPLES_TO_EVALUATE}
```
## Transcribing
PolyAMP can transcribe with melodic and timbre information to create
a multi-track MIDI file, or it can provide single-track MIDI predictions,
which just uses the Melodic Model.
### Single-track MIDI
```bash
python polyamp_runner.py \
    --mode=predict
    --audio_filename=/path/to/audio.file \
    --model_dir=/path/to/models \
    --model_type=MELODIC \
    --load_id={ID_TO_LOAD} \
    --transcribed_file_suffix=-polyamp-single-track \
    --qpm={TEMPO}

```
### Multi-track MIDI
```bash
python polyamp_runner.py \
    --mode=predict
    --audio_filename=/path/to/audio.file \
    --model_dir=/path/to/models \
    --model_type=FULL \
    --load_id=ID_TO_LOAD \
    --transcribed_file_suffix=-polyamp-multi-track
    --qpm={TEMPO}

```
