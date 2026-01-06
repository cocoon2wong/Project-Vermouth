# Project-Vermouth

This is the official developing repo of our paper *Conditioning Trajectory Prediction via Biased Ego Rehearsals*.

## Getting Started

You can clone [this repository](https://github.com/cocoon2wong/Enc) by the following command:

```bash
git clone https://github.com/cocoon2wong/Enc.git
```

Then, run the following command to initialize all submodules:

```bash
git submodule update --init --recursive
```

## Requirements

The code is developed with Python 3.13.  
Additional packages used are included in the `requirements.txt` file.

> [!WARNING]  
> We recommend installing all required Python packages in a virtual environment (like the `conda` environment).  
> Otherwise, there *COULD* be other problems due to the package version conflicts.

Run the following command to install the required packages in your Python environment:

```bash
pip install -r requirements.txt
```

## Installing qpid (CLI support)

This project provides a command-line interface (CLI) for running training and evaluation.

To enable the CLI, install the project:

```bash
pip install -e ./qpid
```

> [!NOTE]
> `-e` means *editable*:  
> any changes you make to the source code will take effect immediately, without reinstalling.

After installation, you can verify:

```bash
qpid --help
```

You can then run experiments simply with:

```bash
qpid --model enc --split zara1 ...
```

(Using `python -m qpid ...` remains fully supported and behaves the same.)

<!-- DO NOT CHANGE THIS LINE -->
---

## Args Used

Please specify your customized args when training or testing your model in the following way:

```bash
python main.py --ARG_KEY1 ARG_VALUE2 --ARG_KEY2 ARG_VALUE2 -SHORT_ARG_KEY3 ARG_VALUE3 ...
```

where `ARG_KEY` is the name of args, and `ARG_VALUE` is the corresponding value.
All args and their usages are listed below.

About the `argtype`:

- Args with argtype=`static` can not be changed once after training.
  When testing the model, the program will not parse these args to overwrite the saved values.
- Args with argtype=`dynamic` can be changed anytime.
  The program will try to first parse inputs from the terminal and then try to load from the saved JSON file.
- Args with argtype=`temporary` will not be saved into JSON files.
  The program will parse these args from the terminal at each time.

### Basic Args


<details markdown="1">
<summary markdown="span"><code>--K</code></summary>

The number of multiple generations when testing. This arg only works for multiple-generation models.

- Type=`int`, argtype=`dynamic`
- The default value is `20`.

</details>

<details markdown="1">
<summary markdown="span"><code>--K_train</code></summary>

The number of multiple generations when training. This arg only works for multiple-generation models.

- Type=`int`, argtype=`static`
- The default value is `10`.

</details>

<details markdown="1">
<summary markdown="span"><code>--anntype</code></summary>

Model's predicted annotation type. Can be `'coordinate'` or `'boundingbox'`.

- Type=`str`, argtype=`static`
- The default value is `coordinate`.

</details>

<details markdown="1">
<summary markdown="span"><code>--auto_clear</code></summary>

Controls whether to clear all other saved weights except for the best one. It performs similarly to running `python scripts/clear.py --logs logs`.

- Type=`int`, argtype=`temporary`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--batch_size</code> (short for <code>-bs</code>)</summary>

Batch size when implementation.

- Type=`int`, argtype=`dynamic`
- The default value is `5000`.

</details>

<details markdown="1">
<summary markdown="span"><code>--compute_loss</code></summary>

Controls whether to compute losses when testing.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--compute_metrics_with_types</code></summary>

Controls whether to compute metrics separately on different kinds of agents.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--compute_statistical_metrics</code></summary>

(bool) Choose whether to compute metrics (ADE/FDE) as `mean $\pm$ std`.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--dataset</code></summary>

Name of the video dataset to train or evaluate. For example, `'ETH-UCY'` or `'SDD'`. NOTE: DO NOT set this argument manually.

- Type=`str`, argtype=`static`
- The default value is `Unavailable`.

</details>

<details markdown="1">
<summary markdown="span"><code>--down_sampling_rate</code></summary>

Selects whether to down-sample from multiple-generated predicted trajectories. This arg only works for multiple-generative models.

- Type=`float`, argtype=`temporary`
- The default value is `1.0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_results</code> (short for <code>-dr</code>)</summary>

Controls whether to draw visualized results on video frames. Accept the name of one video clip. The codes will first try to load the video file according to the path saved in the `plist` file (saved in `dataset_configs` folder), and if it loads successfully it will draw the results on that video, otherwise it will draw results on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_results` if `draw_results != 'null'`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_videos</code></summary>

Controls whether to draw visualized results on video frames and save them as images. Accept the name of one video clip. The codes will first try to load the video according to the path saved in the `plist` file, and if successful it will draw the visualization on the video, otherwise it will draw on a blank canvas. Note that `test_mode` will be set to `'one'` and `force_split` will be set to `draw_videos` if `draw_videos != 'null'`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--epochs</code></summary>

Maximum training epochs.

- Type=`int`, argtype=`static`
- The default value is `500`.

</details>

<details markdown="1">
<summary markdown="span"><code>--experimental</code></summary>

NOTE: It is only used for code tests.

- Type=`bool`, argtype=`temporary`
- The default value is `False`.

</details>

<details markdown="1">
<summary markdown="span"><code>--feature_dim</code></summary>

Feature dimensions that are used in most layers.

- Type=`int`, argtype=`static`
- The default value is `128`.

</details>

<details markdown="1">
<summary markdown="span"><code>--force_anntype</code></summary>

Assign the prediction type. It is now only used for silverballers models that are trained with annotation type `coordinate` but to be tested on datasets with annotation type `boundingbox`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--force_clip</code></summary>

Force test video clip (ignore the train/test split). It only works when `test_mode` has been set to `one`. .

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--force_dataset</code></summary>

Force test dataset (ignore the train/test split). It only works when `test_mode` has been set to `one`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--force_split</code></summary>

Force test dataset (ignore the train/test split).  It only works when `test_mode` has been set to `one`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--gpu</code></summary>

Speed up training or test if you have at least one NVidia GPU.  If you have no GPUs or want to run the code on your CPU,  please set it to `-1`. NOTE: It only supports training or testing on one GPU.

- Type=`str`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--help</code> (short for <code>-h</code>)</summary>

Print help information on the screen.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--input_pred_steps</code></summary>

Indices of future time steps that are used as extra model inputs. It accepts a string that contains several integer numbers separated with `'_'`. For example, `'3_6_9'`. It will take the corresponding ground truth points as the input when  training the model, and take the first output of the former network as this input when testing the model. Set it to `'null'` to disable these extra model inputs.

- Type=`str`, argtype=`static`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--interval</code></summary>

Time interval of each sampled trajectory point.

- Type=`float`, argtype=`static`
- The default value is `0.4`.

</details>

<details markdown="1">
<summary markdown="span"><code>--load</code> (short for <code>-l</code>)</summary>

Folder to load model weights (to test). If it is set to `null`, the training manager will start training new models according to other reveived args. NOTE: Leave this arg to `null` when training new models.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--load_epoch</code></summary>

Load model weights that is saved after specific training epochs. It will try to load the weight file in the `load` dir whose name is end with `_epoch${load_epoch}`. This arg only works when the `auto_clear` arg is disabled (by passing `--auto_clear 0` when training). Set it to `-1` to disable this function.

- Type=`int`, argtype=`temporary`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--load_part</code></summary>

Choose whether to load only a part of the model weights if the `state_dict` of the saved model and the model in the code do not match.

*IMPORTANT NOTE*: This arg is only used for some ablation experiments. It MAY lead to incorrect predictions or metrics.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--log_dir</code></summary>

Folder to save training logs and model weights. Logs will save at `${save_base_dir}/${log_dir}`. DO NOT change this arg manually. (You can still change the saving path by passing the `save_base_dir` arg.).

- Type=`str`, argtype=`static`
- The default value is `Unavailable`.

</details>

<details markdown="1">
<summary markdown="span"><code>--loss_weights</code></summary>

Configure the agent-wise loss weights. It now only supports the dataset-clip-wise re-weight.

- Type=`str`, argtype=`dynamic`
- The default value is `{}`.

</details>

<details markdown="1">
<summary markdown="span"><code>--lr</code> (short for <code>-lr</code>)</summary>

Learning rate.

- Type=`float`, argtype=`static`
- The default value is `0.001`.

</details>

<details markdown="1">
<summary markdown="span"><code>--macos</code></summary>

(Experimental) Choose whether to enable the `MPS (Metal Performance Shaders)` on Apple platforms (instead of running on CPUs).

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--max_agents</code></summary>

Max number of agents to predict per frame. It only works when `model_type == 'frame-based'`.

- Type=`int`, argtype=`dynamic`
- The default value is `50`.

</details>

<details markdown="1">
<summary markdown="span"><code>--model</code></summary>

The model type used to train or test.

- Type=`str`, argtype=`static`
- The default value is `none`.

</details>

<details markdown="1">
<summary markdown="span"><code>--model_name</code></summary>

Customized model name.

- Type=`str`, argtype=`static`
- The default value is `model`.

</details>

<details markdown="1">
<summary markdown="span"><code>--model_type</code></summary>

Model type. It can be `'agent-based'` or `'frame-based'`.

- Type=`str`, argtype=`static`
- The default value is `agent-based`.

</details>

<details markdown="1">
<summary markdown="span"><code>--noise_depth</code></summary>

Depth of the random noise vector.

- Type=`int`, argtype=`static`; also: `--depth`
- The default value is `16`.

</details>

<details markdown="1">
<summary markdown="span"><code>--obs_frames</code> (short for <code>-obs</code>)</summary>

Observation frames for prediction.

- Type=`int`, argtype=`static`
- The default value is `8`.

</details>

<details markdown="1">
<summary markdown="span"><code>--output_pred_steps</code></summary>

Indices of future time steps to be predicted. It accepts a string that contains several integer numbers separated with `'_'`. For example, `'3_6_9'`. Set it to `'all'` to predict points among all future steps.

- Type=`str`, argtype=`static`; also: `--key_points`
- The default value is `all`.

</details>

<details markdown="1">
<summary markdown="span"><code>--pmove</code></summary>

(Pre/post-process Arg) Index of the reference point when moving trajectories.

- Type=`int`, argtype=`static`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--pred_frames</code> (short for <code>-pred</code>)</summary>

Prediction frames.

- Type=`int`, argtype=`static`
- The default value is `12`.

</details>

<details markdown="1">
<summary markdown="span"><code>--preprocess</code></summary>

Controls whether to run any pre-process before the model inference. It accepts a 3-bit-like string value (like `'111'`): - The first bit: `MOVE` trajectories to (0, 0); - The second bit: re-`SCALE` trajectories; - The third bit: `ROTATE` trajectories.

- Type=`str`, argtype=`static`
- The default value is `100`.

</details>

<details markdown="1">
<summary markdown="span"><code>--restore</code></summary>

Path to restore the pre-trained weights before training. It will not restore any weights if `args.restore == 'null'`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--restore_args</code></summary>

Path to restore the reference args before training. It will not restore any args if `args.restore_args == 'null'`.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--save_base_dir</code></summary>

Base folder to save all running logs.

- Type=`str`, argtype=`static`
- The default value is `./logs`.

</details>

<details markdown="1">
<summary markdown="span"><code>--split</code> (short for <code>-s</code>)</summary>

The dataset split that used to train and evaluate.

- Type=`str`, argtype=`static`
- The default value is `zara1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--start_test_percent</code></summary>

Set when (at which epoch) to start validation during training. The range of this arg should be `0 <= x <= 1`.  Validation may start at epoch `args.epochs * args.start_test_percent`.

- Type=`float`, argtype=`temporary`
- The default value is `0.0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--step</code></summary>

Frame interval for sampling training data.

- Type=`float`, argtype=`dynamic`
- The default value is `1.0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--test_mode</code></summary>

Test settings. It can be `'one'`, `'all'`, or `'mix'`. When setting it to `one`, it will test the model on the `args.force_split` only; When setting it to `all`, it will test on each of the test datasets in `args.split`; When setting it to `mix`, it will test on all test datasets in `args.split` together.

- Type=`str`, argtype=`temporary`
- The default value is `mix`.

</details>

<details markdown="1">
<summary markdown="span"><code>--test_step</code></summary>

Epoch interval to run validation during training.

- Type=`int`, argtype=`temporary`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--update_saved_args</code></summary>

Choose whether to update (overwrite) the saved arg files or not.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--verbose</code> (short for <code>-v</code>)</summary>

Controls whether to print verbose logs and outputs to the terminal.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

### Visualization Args


<details markdown="1">
<summary markdown="span"><code>--distribution_steps</code></summary>

Controls which time step(s) should be considered when visualizing the distribution of forecasted trajectories. It accepts one or more integer numbers (started with 0) split by `'_'`. For example, `'4_8_11'`. Set it to `'all'` to show the distribution of all predictions.

- Type=`str`, argtype=`temporary`
- The default value is `all`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_distribution</code> (short for <code>-dd</code>)</summary>

Controls whether to draw distributions of predictions instead of points. If `draw_distribution == 0`, it will draw results as normal coordinates; If `draw_distribution == 1`, it will draw all results in the distribution way, and points from different time steps will be drawn with different colors.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_exclude_type</code></summary>

Draw visualized results of agents except for user-assigned types. If the assigned types are `"Biker_Cart"` and the `draw_results` or `draw_videos` is not `"null"`, it will draw results of all types of agents except "Biker" and "Cart". It supports partial match, and it is case-sensitive.

- Type=`str`, argtype=`temporary`
- The default value is `null`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_extra_outputs</code></summary>

Choose whether to draw (put text) extra model outputs on the visualized images.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_full_neighbors</code></summary>

Choose whether to draw the full observed trajectories of all neighbor agents or only the last trajectory point at the current observation moment.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_index</code></summary>

Indexes of test agents to visualize. Numbers are split with `_`. For example, `'123_456_789'`.

- Type=`str`, argtype=`temporary`
- The default value is `all`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_lines</code></summary>

Choose whether to draw lines between each two 2D trajectory points.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_on_empty_canvas</code></summary>

Controls whether to draw visualized results on the empty canvas instead of the actual video.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_with_plt</code></summary>

(bool) Choose whether to use PLT as the preferred method for visualizing trajectories (on the empty canvas). It will try to visualize all points on the scene images if this arg is not enabled. .

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

### Encore Args


<details markdown="1">
<summary markdown="span"><code>--Kg</code></summary>

The number of generations when making predictions. It is also the channels of the generating kernel in the proposed reverberation transform.

- Type=`int`, argtype=`static`
- The default value is `20`.

</details>

<details markdown="1">
<summary markdown="span"><code>--T</code> (short for <code>-T</code>)</summary>

Transform type used to compute trajectory spectrums.

It could be: - `none`: no transformations; - `haar`: haar wavelet transform; - `db2`: DB2 wavelet transform.

- Type=`str`, argtype=`static`
- The default value is `haar`.

</details>

<details markdown="1">
<summary markdown="span"><code>--ego_capacity</code></summary>

The number of neighbors (`N`) to be "well-forecasted" by the ego predictor. When there are more numbers of neighbors than this value in the scene, the ego predictor will choose the most-`N` closed neighbors in relation to the ego agent to run the full-size prediction, while other neighbors will be forecasted using a simple linear predictor.

**Ablation Settings:** Note that the full-size Transformer-based ego predictor will be constructed and used for prediction only when `N > 0`. A linear predictor will be used for all neighbors when `N` is set to `0`.

- Type=`int`, argtype=`dynamic`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--ego_loss_rate</code></summary>

Loss weight of the EgoLoss when training.

- Type=`float`, argtype=`static`
- The default value is `0.6`.

</details>

<details markdown="1">
<summary markdown="span"><code>--ego_t_f</code></summary>

Output length of the ego predicotr.

- Type=`int`, argtype=`static`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--ego_t_h</code></summary>

Input length of the ego predicotr.

- Type=`int`, argtype=`static`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--encode_agent_types</code></summary>

Choose whether to encode the type name of each agent. It is mainly used in multi-type-agent prediction scenes, providing a unique type-coding for each type of agents when encoding their trajectories.

- Type=`int`, argtype=`static`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--insights</code></summary>

The number of "insights" (`I`) in the ego predictor. The full-size ego predictor will forecast `I` short-term trajectories for each neighbor within its capacity.

- Type=`int`, argtype=`static`
- The default value is `5`.

</details>

<details markdown="1">
<summary markdown="span"><code>--partitions</code></summary>

The number of partitions when computing the angle-based feature. It is only used when modeling social interactions.

- Type=`int`, argtype=`static`
- The default value is `-1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--use_intention_predictor</code></summary>

**Ablation Settings:** (bool) Choose whether to use the intention prediction as one of the model predictions.

- Type=`int`, argtype=`static`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--use_linear</code></summary>

**Ablation Settings:** (bool) Choose whether to use the linear prediction as the base of all other predictions.

- Type=`int`, argtype=`static`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--use_social_predictor</code></summary>

**Ablation Settings:** (bool) Choose whether to use the social prediction as one of the model predictions.

- Type=`int`, argtype=`static`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--vis_ego_predictor</code></summary>

Choose whether to visualize trajectories forecasted by the ego predictior. It accepts three values:

- `0`: Do nothing; - `1`: Visualize ego predictor's all predictions; - `2`: Visualize ego predictor's mean predicton for each neighbor.

NOTE that this arg only works in the *Playground* mode, or the program will be killed immediately.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

### Playground Args


<details markdown="1">
<summary markdown="span"><code>--clip</code></summary>

The video clip to run this playground.

- Type=`str`, argtype=`temporary`
- The default value is `zara1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--compute_social_mod</code></summary>

(bool) Choose whether to enable the computing of social modifications.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--default_agent</code></summary>

Set the default index of agent to be predicted.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--do_not_draw_neighbors</code></summary>

(bool) Choose whether to draw neighboring-agents' trajectories.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--draw_seg_map</code></summary>

(bool) Choose whether to draw segmentation maps on the canvas.

- Type=`int`, argtype=`temporary`
- The default value is `1`.

</details>

<details markdown="1">
<summary markdown="span"><code>--lite</code></summary>

(bool) Choose whether to show the lite-version's visualization window.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--physical_manual_neighbor_mode</code></summary>

Mode for the manual neighbor on segmentation maps. - Mode `1`: Add obstacles to the given position; - Mode `0`: Set areas to be walkable.

- Type=`float`, argtype=`temporary`
- The default value is `1.0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--points</code></summary>

The number of points to simulate the trajectory of manual neighbor. It only accepts `2` or `3`.

- Type=`int`, argtype=`temporary`
- The default value is `2`.

</details>

<details markdown="1">
<summary markdown="span"><code>--save_full_outputs</code></summary>

(bool) Choose whether to save all outputs as images.

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>

<details markdown="1">
<summary markdown="span"><code>--show_manual_neighbor_boxes</code></summary>

(Working in process)

- Type=`int`, argtype=`temporary`
- The default value is `0`.

</details>
<!-- DO NOT CHANGE THIS LINE -->
