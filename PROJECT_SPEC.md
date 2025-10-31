# TASK LIST FOR STREAMLIT MULTICLASS THRESHOLD TUNER
================================================================================

## Setup & Environment
----------------------------------------


## Core Functions
----------------------------------------

### Task 3: Implement activation functions module
**Priority:** HIGH
**Dependencies: Tasks [1]**


In utils/activations.py, implement:
1. softmax(logits) - vectorized across class dimension
2. sigmoid(x) - elementwise sigmoid
3. sigmoid_5(x) - custom sigmoid with scale factor 5: 1/(1 + exp(-x/5))
4. apply_activation(scores, activation_type) - wrapper function that applies the selected activation or returns raw scores
All functions must handle numpy arrays efficiently and avoid loops over samples.


### Task 4: Implement data loading and validation module
**Priority:** HIGH
**Dependencies: Tasks [1]**


In utils/data_io.py, implement:
1. load_csv(file_path) - load CSV with configurable delimiter
2. validate_data(df) - check for required columns (true_label), infer class columns
3. prepare_score_matrix(df, classes) - convert to wide format [n_samples, n_classes] matrix
4. handle_missing_scores(scores) - impute NaNs to -inf before activation
5. get_majority_class(y_true) - compute and cache the mode of true labels
Handle both wide format (logit_{class} columns) and narrow format (true_label, predicted_category, logit_score).


### Task 5: Implement threshold management module
**Priority:** HIGH
**Dependencies: Tasks [3, 4]**


In utils/thresholds.py, implement:
1. predict_with_thresholds(scores, classes, thresholds, activation, fallback_class) - main prediction function
2. compute_optimal_thresholds_youden(y_true, y_scores, classes) - find J-optimal thresholds per class
3. apply_thresholds(scores, thresholds, classes) - create boolean mask of exceeding thresholds
4. select_predicted_class(masked_scores, fallback_class) - pick argmax among exceeding or fallback
Ensure vectorized operations for 50k+ rows performance.


## Metrics
----------------------------------------

### Task 6: Implement metrics computation module
**Priority:** HIGH
**Dependencies: Tasks [5]**


In utils/metrics.py, implement:
1. compute_classification_metrics(y_true, y_pred) - return accuracy, macro precision/recall/F1 using sklearn
2. per_class_roc_and_j(y_true, y_scores, classes) - compute ROC curves, AUC, and Youden's J per class
3. compute_micro_macro_auc(y_true, y_scores, classes) - compute micro and macro averaged AUC
4. create_metrics_summary(y_true, y_pred, y_scores, classes) - aggregate all metrics into a dictionary
Handle edge cases: classes with zero positive samples, NaN/inf values.


## Visualization
----------------------------------------

### Task 7: Implement confusion matrix visualization
**Priority:** HIGH
**Dependencies: Tasks [6]**


In utils/plots.py, implement plot_confusion_matrix_raw():
- Use sklearn.metrics.confusion_matrix with labels=classes
- Style with: cmap='YlGnBu', PowerNorm(gamma=0.2), annot=True, fmt='d'
- Set monospace font via rcParams
- Add title with F1-Macro score
- Rotate x-labels 45 degrees (ha='right'), y-labels 0 degrees
- Set figure size (12, 10) with tight layout
- Save at 300 dpi if output path provided
- Return figure object for Streamlit display


### Task 8: Implement ROC curve visualizations
**Priority:** MEDIUM
**Dependencies: Tasks [6, 7]**


In utils/plots.py, add:
1. plot_per_class_roc(roc_data, classes, filter_classes=None) - create individual or multi-panel ROC plots
2. format_roc_figure() - apply monospace font and consistent styling
3. create_inline_roc_display() - prepare figures for st.pyplot() display
Support both individual plots per class and combined multi-panel view.
Save figures to outputs/figures/ with snake_case naming.


## UI Components
----------------------------------------

### Task 10: Create CSS styling and monospace font configuration
**Priority:** MEDIUM
**Dependencies: Tasks [1]**


In assets/styles.css, create CSS for:
1. Monospace font for all Streamlit elements
2. Sidebar styling for controls
3. Main area layout optimization
4. Table formatting for Youden's J display
Include CSS injection function for Streamlit to apply monospace globally.


### Task 11: Build sidebar controls in Streamlit
**Priority:** HIGH
**Dependencies: Tasks [10]**


In main.py, implement sidebar with:
1. File uploader (accept CSV)
2. Activation function dropdown (none, softmax, sigmoid, sigmoid_5)
3. Global threshold slider (default 0.5 for sigmoid variants, 0.0 for raw)
4. Dynamic per-class threshold sliders (auto-populated from data)
5. Auto-threshold toggle (global and per-class)
6. Class filter multiselect
7. Download buttons section
Use st.sidebar for all controls.


### Task 12: Build main display area
**Priority:** HIGH
**Dependencies: Tasks [11]**


In main.py, implement main area with:
1. Key metrics display (accuracy, macro precision/recall/F1)
2. Confusion matrix plot (inline with st.pyplot)
3. Per-class ROC curves display
4. Micro/Macro AUC text display
5. Youden's J table with columns [class, optimal_threshold, J, TPR, FPR, AUC]
6. Optional PR curves at bottom
Use columns/containers for layout organization.


## Exports
----------------------------------------

### Task 9: Implement export functionality
**Priority:** MEDIUM
**Dependencies: Tasks [7, 8]**


In utils/exports.py, implement:
1. save_classification_report(report, format='json') - save as JSON and CSV
2. save_confusion_matrix_image(fig, timestamp) - save PNG at 300dpi
3. save_roc_images(figures, classes, timestamp) - save individual ROC PNGs
4. save_predictions_csv(df, thresholds, timestamp) - save predictions with scores and thresholds
5. generate_timestamp() - create consistent timestamp format for filenames
All files use snake_case naming convention.


## Integration
----------------------------------------

### Task 13: Implement caching and state management
**Priority:** HIGH
**Dependencies: Tasks [11, 12]**


Add Streamlit caching:
1. @st.cache_data for loaded CSV data
2. @st.cache_data for computed metrics (keyed by activation + thresholds)
3. @st.cache_resource for heavy computations
4. Session state management for threshold values
5. Cache invalidation logic when parameters change
Optimize for 50k row datasets.


### Task 14: Wire up live updates and interactivity
**Priority:** HIGH
**Dependencies: Tasks [13]**


Connect all components for live updates:
1. Threshold slider changes trigger prediction recomputation
2. Activation change triggers full pipeline refresh
3. Class filter updates visualizations without recomputing predictions
4. Auto-threshold toggle updates slider values
5. Download buttons generate artifacts on-demand
Ensure smooth performance with proper caching.


### Task 19: Handle edge cases and error messages
**Priority:** MEDIUM
**Dependencies: Tasks [14]**


Implement proper error handling for:
1. Missing required columns in CSV
2. Classes with zero support in y_true
3. Activation functions producing NaN/inf
4. Missing class scores (warn and use defaults)
5. File size limits and memory issues
Display user-friendly error messages in Streamlit.


## Testing
----------------------------------------

### Task 15: Create synthetic data generator
**Priority:** MEDIUM
**Dependencies: Tasks [4]**


In tests/generate_synthetic_data.py:
1. Generate N samples (configurable, default 5000)
2. K classes (configurable, default 10)
3. Raw logits ~ Normal(0, 3)
4. True labels via categorical sampling
5. Save as CSV in both wide and narrow formats
6. Include edge cases (imbalanced classes, near-threshold scores)


### Task 16: Write unit tests for core functions
**Priority:** MEDIUM
**Dependencies: Tasks [3, 5, 6]**


Create unit tests for:
1. test_activations.py - test all activation functions with edge cases
2. test_thresholds.py - test threshold application and prediction logic
3. test_metrics.py - test ROC, AUC, Youden's J computation
Include tests for NaN handling, empty classes, and performance with large arrays.


## Optimization
----------------------------------------

### Task 17: Add PR curves and additional metrics
**Priority:** LOW
**Dependencies: Tasks [12]**


Optional enhancements:
1. Implement precision-recall curves per class
2. Calculate macro/weighted AUPRC
3. Add interactive Plotly visualizations
4. Create comparative views for different activation functions
Display at bottom of main area.


## Documentation
----------------------------------------

### Task 18: Create README and usage documentation
**Priority:** LOW
**Dependencies: Tasks [14]**


Write documentation including:
1. README.md with setup instructions
2. Data format requirements and examples
3. UI walkthrough with screenshots
4. Performance benchmarks for different data sizes
5. Troubleshooting guide for common issues


## Final Integration
----------------------------------------

### Task 20: End-to-end testing and validation
**Priority:** HIGH
**Dependencies: Tasks [19]**


Perform complete testing:
1. Load synthetic data of various sizes (1k, 10k, 50k rows)
2. Test all activation functions
3. Verify threshold adjustments update metrics correctly
4. Validate all export functions produce correct files
5. Check performance meets requirements (live updates, <1s response)
6. Test with real-world CSV matching the specified format
Fix any integration issues discovered.


================================================================================
## RECOMMENDED IMPLEMENTATION ORDER:

1. **Phase 1 - Foundation (Tasks 1-4):** Set up project structure, environment, and core data handling
2. **Phase 2 - Core Logic (Tasks 3-6):** Implement activation functions, thresholds, and metrics
3. **Phase 3 - Visualization (Tasks 7-8, 10):** Create confusion matrix and ROC visualizations with styling
4. **Phase 4 - UI Development (Tasks 11-12):** Build Streamlit interface with sidebar and main display
5. **Phase 5 - Integration (Tasks 13-14, 19):** Wire up interactivity, caching, and error handling
6. **Phase 6 - Exports & Testing (Tasks 9, 15-16, 20):** Add export functionality and comprehensive testing
7. **Phase 7 - Polish (Tasks 17-18):** Add optional enhancements and documentation

**Critical Path:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 11 → 12 → 13 → 14 → 20
