## PolyAMP: Poly-Instrumental Audio to Midi Prediction

#Training
To train, it's helpful to specify `--model_id` so that checkpoints for
that id are saved in one place. If a saved id is set with `--load_id`,
the latest checkpoint from training that model will be loaded.
Otherwise, `--load_id` defaults to "`*`", and whatever is the most recent
checkpoint will be loaded.

Additionally, hyperparameters can be overridden by specifying `--hparams`.
For example:\
`--hparams={\"frames_true_weighing\":3,\"frame_prediction_threshold\":0.4}`


- Melodic Model
```bash
python trainer.py \
    --examples_path=/path/to/maestro/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_id=ANY_ID_STRING \
    --load_id=ID_TO_LOAD
```
- Timbre Model
```bash
python trainer.py \
    --examples_path=/path/to/nsynth/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=TIMBRE \
    --dataset_type=nsynth \
    --model_id=ANY_ID_STRING \
    --load_id=ID_TO_LOAD
```
```bash
python trainer.py \
    --examples_path=/path/to/slakh/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=TIMBRE \
    --dataset_type=slakh \
    --model_id=ANY_ID_STRING \
    --load_id=ID_TO_LOAD
```
- Full Model
```bash
python trainer.py \
    --examples_path=/path/to/maestro/train.tfrecord-*,/slakh/train.tfrecord-*,/custom/train.tfrecord-* \
    --nsynth_examples_path=/path/to/nsynth/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=FULL \
    --model_id=ANY_ID_STRING \
    --load_id=ID_TO_LOAD
```

## Evaluating
- Melodic Model
```bash
python trainer.py \
    --examples_path=/path/to/maestro/train.tfrecord-*,/slakh/train.tfrecord-*,/custom/train.tfrecord-* \
    --nsynth_examples_path=/path/to/nsynth/train.tfrecord-* \
    --model_dir=/path/to/models \
    --model_type=FULL \
    --model_id=ANY_ID_STRING \
    --load_id=ID_TO_LOAD
```
- Timbre Model

- Full Model

## Transcribing
- Single-track MIDI

- Multi-track MIDI

