# Project Summary - Methods, Architecture, and Results

**Austin Samuel Qiu · August–September 2026**  
**Working Summary & Review**

## Project Goal

I started this project to gain more experience in implementing and evaluating XAI methods, while keeping the experiment small enough to understand each part of the pipeline, and familiarizing myself with the workflow of conducting research. The main question was whether local explanations could help identify a model that relies on an artificial shortcut feature instead of the intended time-series signal.

The implemented experiment compares linear regression, kernel ridge regression (kRR), and MLP models on synthetic data, both with and without a shortcut. LRP is then applied to the MLP, followed by intervention validation where only the shortcut feature is changed. The additional ReLU, LogSigmoid, and GELU comparison looks more closely at the flat prediction region found in the earlier ReLU results.

This document summarizes the methods and results. The [README](../README.md) contains the setup and run commands, while the [research logs](research_logs.md) contain the earlier implementation notes and interpretations throughout the working process. The LaTeX source and PDF can be linked here once the report is ready.

## Data and Shortcut Feature

The synthetic target combines a linear trend, annual seasonality that grows with time, and Gaussian noise. The dataset contains 3,650 observations:

```text
f_linear(t)           = 0.02 × t
annual_seasonality(t) = sin(π × ((t mod 365) − 182) / 182)
y(t)                 = f_linear(t) + (t / 400) × annual_seasonality(t) + noise
noise                ~ Normal(0, 0.05²)
```

The baseline input contains `time`, `f_linear`, and `annual_seasonality`. The model receives these features for each time index, rather than a window of previous target values. The test split therefore checks how the model predicts over a later time interval. Note that the features `time` and `f_linear` contain the same information because one is a fixed multiple of the other. Their separate relevance shares should be interpreted with that redundancy in mind.

| Split | Time indices | Observations |
|---|---|---:|
| Train | 0–2554 | 2,555 |
| Validation | 2555–3101 | 547 |
| Test | 3102–3649 | 548 |

The splits preserve time order. `StandardScaler` is fitted on training inputs and applied to the other splits; the target values remain unscaled. The baseline and shortcut datasets use the same data seed, 42, so their targets and normal features match.

For the shortcut, I opted to fit a degree-20 Chebyshev polynomial to the noisy targets from the train and validation splits. The polynomial is evaluated at every time index and added as a fourth feature, `shortcut_polynomial`. It can appear useful within the fitted range, but produces extreme values when extrapolated into the test split.

This deliberately uses validation labels when constructing the feature. As such, the shortcut validation score cannot be treated as an independent generalization result. Test labels are not included in the polynomial fit. Earlier notes referred to interpolation and Runge's phenomenon; the implemented method is a least-squares polynomial fit followed by extrapolation beyond the fitted interval into the test split.

## Model Implementation

The comparison follows the same basic steps for baseline and shortcut data:

```text
Generate data → split by time → scale input features
    → train linear regression, kRR, and MLP
    → compare predictions and metrics
    → calculate MLP relevance and apply shortcut interventions
    → inspect hidden activations and save results and figures
```

| Model | Implemented setup |
|---|---|
| Linear regression | Ordinary least squares with an intercept |
| kRR | RBF kernel, with alpha and gamma selected using validation RMSE |
| MLP | 3 or 4 inputs → 8 hidden units → 4 hidden units → 1 output |

For kRR, the grid search tests `alpha, gamma ∈ {0.001, 0.005, 0.01, 0.1, 1.0}`. The selected baseline settings are `alpha=0.001, gamma=0.001`; the shortcut settings are `alpha=0.005, gamma=0.001`. These are the best values within the tested grid.

For the MLP, an activation function follows each hidden linear layer, while the output layer stays linear. Both hidden layers use the selected activation: ReLU, LogSigmoid, or GELU. Training uses Adam, MSE loss, learning rate 0.01, batches of 32, and a maximum of 50 epochs. Early stopping checks validation loss with a patience value of 5 and a minimum improvement of 0.0001, then restores the best weights recorded. Seed 42 is used for initialization and batch shuffling.

The activation comparison uses the same data and training settings, but each model is trained separately. Early stopping therefore gives different completed epoch counts:

| Activation | Saved result | Completed epochs: base / shortcut |
|---|---|---:|
| ReLU | [2026_09_13_15_46_44_relu](../results/model_comparison_2026_09_13_15_46_44_relu.json) | 23 / 8 |
| LogSigmoid | [2026_09_13_15_47_08_lgsigmoid](../results/model_comparison_2026_09_13_15_47_08_lgsigmoid.json) | 50 / 42 |
| GELU | [2026_09_13_15_47_33_gelu](../results/model_comparison_2026_09_13_15_47_33_gelu.json) | 14 / 27 |

## LRP and Intervention Validation

The LRP implementation uses Zennit's `Gradient` attributor with the `EpsilonPlus` composite and `epsilon=1e-6`. The output attribution is initialized with `torch.ones_like`, which gives each scalar output a unit seed. Consequently, the relevance values are not assumed to sum to the prediction in target units.

The recorded propagation rules are `Epsilon` for the linear layers and `Pass` for each of the three activation types.

Relevance summaries are recorded differently within the JSON file and the visualizations. The JSON summary first takes the mean absolute relevance of each feature and then normalizes across features. The heatmaps, line maps, and scatter colors instead normalize each sample separately:

```text
Aggregate share = mean absolute relevance of feature / total mean absolute relevance
Local share     = absolute relevance of feature / total absolute relevance of that sample
```

The local denominator is limited to at least `1e-12` to handle zero relevance. Note that due to the construction of each share, a high share can occur even when the total relevance is small, so the activation diagnostics also show the raw magnitudes.

The intervention functions change only the standardized shortcut column on the test split. The trained model, other features, and target values remain unchanged.

| Condition | Change to the shortcut |
|---|---|
| Normal | Keep the original extrapolated values |
| Zeroed | Set scaled values to zero, equivalent to the shortcut's training mean in raw units |
| Noise | Replace with standard normal noise using seed 42 |
| Reversed | Multiply scaled values by −1; this reflects raw values around their training mean |
| Permuted | Shuffle test shortcut values across rows using seed 42 |

## Results and Interpretation

### Model Comparison

The following table records RMSE, rounded to four decimal places. MAE and R² are also saved in the linked JSON files. 

| Model / activation | Baseline test | Shortcut validation | Shortcut test |
|---|---:|---:|---:|
| Linear regression | 3.6760 | 2.5168 | 28,532.6805 |
| RBF kRR | 0.5799 | 0.1997 | 63.6167 |
| MLP, ReLU | 1.9101 | 0.9532 | 200,119.5748 |
| MLP, LogSigmoid | 1.1210 | 0.3624 | 99,561.6661 |
| MLP, GELU | 1.8699 | 0.7848 | 186,181.4639 |

The baseline kRR model performs best among the tested models, while LogSigmoid gives the lowest baseline error among the MLPs. All shortcut models perform much worse on the test split than their baseline versions. This matches the intended failure of the shortcut: its relationship with the target does not hold outside the fitted interval.

### Intervention Results

| Test condition | ReLU RMSE | LogSigmoid RMSE | GELU RMSE |
|---|---:|---:|---:|
| Normal | 200,119.5748 | 99,561.6661 | 186,181.4639 |
| Zeroed | 15.9427 | 9.6873 | 11.4421 |
| Noise | 16.8427 | 10.2537 | 12.0891 |
| Reversed | 685.3291 | 33,107.6201 | 13,773.8993 |
| Permuted | 200,121.0512 | 99,563.0523 | 186,183.1371 |

With the original test shortcut, its aggregate relevance share is **76.04% for ReLU, 70.63% for LogSigmoid, and 76.39% for GELU**. In each case it receives the largest share. Setting it to zero or replacing it with noise substantially reduces RMSE, although the result is still worse than the corresponding baseline MLP.

The main interpretation is that the shortcut model has learned a harmful dependency on this feature. Since the original shortcut is already unreliable during testing, removing it can improve performance. Intervention validation here therefore checks whether changing the suspected feature changes model behavior. Similarly, a zero shortcut relevance share after zeroing the input is not sufficient by itself to validate LRP.

LogSigmoid performs better in the normal, zeroed, and noise conditions, but its reversed-shortcut result is much worse than ReLU's. GELU also retains a large test-time failure. These results give no reason to conclude that changing activation functions solves the shortcut problem.

### ReLU Flatline

The earlier ReLU plots showed an almost constant prediction around time 3190–3400, accompanied by very low relevance. The additional measurements examine the values before and after each hidden activation to check what happens in that interval. The interval was selected from the earlier plots before the new comparison runs, and contains 211 test samples.

| Measurement in that interval | ReLU | LogSigmoid | GELU |
|---|---:|---:|---:|
| Rows with all four second-hidden-layer outputs exactly zero | 211 / 211 | 0 / 211 | 0 / 211 |
| Rows with total absolute input relevance exactly zero | 211 / 211 | 0 / 211 | 0 / 211 |
| Prediction range | 0.5027–0.5027 | −281.9319 to −0.7304 | −119.0210 to 14.3265 |

For ReLU, every input to the second hidden activation is negative in this interval, ranging from approximately −244.47 to −0.27. ReLU maps all of these values to zero. The output layer therefore receives four zeros, leaving only its bias:

```text
prediction = output weights × [0, 0, 0, 0] + output bias
           = 0.5027097
```

This explains the flatline in this trained model. The raw input relevance is also exactly zero throughout the interval, so the low relevance is not caused only by normalization in the plots.

There is still a distinction between units being inactive in this region and being permanently inactive during training. Two second-layer units, indices 0 and 2, are zero across all train, validation, and test samples when checked with the final weights. Their values were not tracked throughout training, so the results do not establish when they became inactive or whether they stayed inactive during optimization.

LogSigmoid and GELU do not have an entirely zero second hidden layer in this interval. Their predictions vary, and their total input relevance remains nonzero. Some individual units still give exactly zero outputs in floating-point calculations; the saved measurements distinguish exact zeros from small absolute values using a threshold of `1e-8`.

The measurements support the proposed ReLU explanation for this particular flatline. However, the comparison retrains each network, changing its weights and training duration as well as its activation. It does not isolate what would happen if only the activation were replaced in a fixed trained network, and it does not show that the smoother activations produce more faithful explanations.

### Generated Plots

The activation diagnostic figures show predictions, the fraction of hidden units that output exactly zero, and raw relevance over the same test interval. Prediction and relevance use symmetric logarithmic axes to make both small and extreme values visible.

| Evidence | ReLU | LogSigmoid | GELU |
|---|---|---|---|
| Activation diagnostics | [Figure](activation%20diagnostics/activation_diagnostics_2026_09_13_15_46_44_relu.png) | [Figure](activation%20diagnostics/activation_diagnostics_2026_09_13_15_47_08_lgsigmoid.png) | [Figure](activation%20diagnostics/activation_diagnostics_2026_09_13_15_47_33_gelu.png) |
| Clean predictions | [Figure](model%20comparisons/compare_baseline_2026_09_13_15_46_44_relu.png) | [Figure](model%20comparisons/compare_baseline_2026_09_13_15_47_08_lgsigmoid.png) | [Figure](model%20comparisons/compare_baseline_2026_09_13_15_47_33_gelu.png) |
| Shortcut predictions colored by local relevance | [Figure](lrp%20shortcut%20scatters/lrp_scatter_shortcut_normal_2026_09_13_15_46_44_relu.png) | [Figure](lrp%20shortcut%20scatters/lrp_scatter_shortcut_normal_2026_09_13_15_47_08_lgsigmoid.png) | [Figure](lrp%20shortcut%20scatters/lrp_scatter_shortcut_normal_2026_09_13_15_47_33_gelu.png) |
| Local relevance shares over time | [Figure](lrp%20line%20maps/lrp_line_shortcut_2026_09_13_15_46_44_relu.png) | [Figure](lrp%20line%20maps/lrp_line_shortcut_2026_09_13_15_47_08_lgsigmoid.png) | [Figure](lrp%20line%20maps/lrp_line_shortcut_2026_09_13_15_47_33_gelu.png) |

## Going Through the Implementation

The following order follows the additions made for the supplementary activation comparison. These additions were implemented with ChatGPT Work / Codex assistance, as described in the [AI assistance note](#use-of-ai-assistance-in-the-supplementary-work). The credit applies to the new activation option, diagnostics, saving helpers, tests, and their integration, rather than to the entire existing files.

1. [mlp.py](../src/xai_mini_research/models/mlp.py): start with `MLPRegressor`. The activation argument selects the function used after both hidden layers. The existing training procedure is unchanged.
2. [diagnostics.py](../src/xai_mini_research/diagnostics.py): `inspect_hidden_activations` follows the forward pass one layer at a time. `summarize_hidden_layer` then counts zero and near-zero outputs. In these arrays, rows are samples and columns are hidden units.
3. [compare_models.py](../src/xai_mini_research/experiments/compare_models.py): read `main` to follow the data, training, relevance, intervention, and hidden-layer checks. The same checks are applied to baseline and shortcut MLPs.
4. [results.py](../src/xai_mini_research/results.py): the saving helpers keep the trained weights, scaler parameters, and a copy of the source code. These files make it possible to inspect a run again later.
5. [test_activation_diagnostics.py](../tests/test_activation_diagnostics.py): the small examples check activation selection, regional inactivity, and the difference between exact and near-zero values. The simple ReLU example gives a useful case to calculate by hand.

The original data and explanation methods remain in [data.py](../src/xai_mini_research/data.py), [preprocessing.py](../src/xai_mini_research/preprocessing.py), [explain.py](../src/xai_mini_research/explain.py), and [interventions.py](../src/xai_mini_research/interventions.py). The effective experiment settings are in `compare_models.py`; the YAML file is still a setup placeholder.

*Supplementary implementation and walkthrough: ChatGPT Work / Codex — September 13, 2026.*

## Saved Results and Verification

The tables above use the three named September 13 runs. Each JSON records its activation, training settings, completed epochs, library versions, LRP rules, and figure paths. Each matching `results/artifacts_<timestamp>_<activation>/` folder contains saved MLP weights, fitted scaler parameters, intermediate hidden-layer values, and prediction and relevance arrays. The `source.zip` file preserves the code used for that run, including the local changes based on commit `e73c793`.

The current code has since been reorganized and commented for review. The saved experiments and their original source copies remain unchanged. The older [September 04](../results/model_comparison_2026_09_04_00_23_04.json) and [September 08](../results/model_comparison_2026_09_08_07_42_57.json) files remain available too; they lack activation metadata, while the newly labelled ReLU and LogSigmoid runs reproduce their reported predictive and intervention metrics.

All **59 tests passed** after the readability changes. The revised comparison script was also run for all three activations in a temporary directory. Metrics, training losses, saved weights, hidden-layer summaries, and all 53 saved arrays per run matched the original runs exactly. The original result files and figures were left unchanged. These are implementation checks within the same environment, rather than independent repetitions of the study. The [README](../README.md#run-the-current-experiment) gives the activation-specific run commands.

*Supplementary experiment execution, verification, and documentation: ChatGPT Work / Codex — September 13, 2026. The code execution and checks described above were performed through Codex; my own review of the implementation is still in progress.*

## Current Limitations and Next Steps

For now, the experiment provides a small example where LRP and intervention validation together help identify harmful shortcut reliance. It is based on one synthetic construction and one seed. Other limitations include the deliberately leaked validation labels, redundant trend features, extreme extrapolated shortcut values, exploratory reuse of the test results, and limited quantitative validation of the explanations.

The next step is to go through the added implementation and these interpretations before making final revisions for the report. Further experiments with new seeds and fresh evaluation data could test whether the observations hold more broadly. Tracking hidden activations during training could also help answer when nodes become inactive. These are possible follow-up questions rather than conclusions established by the current results.

## Use of AI Assistance in the Supplementary Work

Most of the earlier implementation was completed by me, with AI assistance mainly used for repetitive plotting, some work under time constraints, technical discussion, and suggested tests.

For the supplementary ReLU, LogSigmoid, and GELU exploration, I chose to use ChatGPT Work for implementation and documentation assistance, with the code changes and execution carried out through Codex. This exploration extends beyond the original goal of implementing the controlled shortcut experiment, LRP, and intervention validation. Given the time constraints, I wanted to focus primarily on the mathematical and scientific implications of the supplementary results, while using AI assistance to implement the additional measurements and organize their outputs.

This assistance covered activation selection, hidden-layer diagnostics, saving model weights and experiment details, integration into the comparison script, additional tests, execution and verification of the comparison runs, and preparation of the implementation and verification sections of this summary. The code was subsequently revised into a style closer to the existing project to make it easier for me to work through.

I have reviewed these additions before making final revisions and writing the final report. Understanding the implementation and taking responsibility for the interpretation remain part of that review. The automated checks described above do not replace my review or constitute independent scientific validation.
