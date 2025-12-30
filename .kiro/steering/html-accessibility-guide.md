---
inclusion: always
---

# HTML Accessibility Guide

This document outlines accessibility standards for the ML Training Platform frontend, ensuring compliance with WCAG 2.1 AA guidelines and creating an inclusive user experience for all users, including those using assistive technologies.

## Quick Reference Checklist

### Essential Accessibility Requirements

- [ ] All form inputs have associated labels
- [ ] Color contrast meets 4.5:1 ratio for normal text, 3:1 for large text
- [ ] All interactive elements are keyboard accessible
- [ ] Focus indicators are visible and clear
- [ ] Images and charts have descriptive alt text or ARIA labels
- [ ] Error messages are announced to screen readers
- [ ] Dynamic content changes are announced via live regions
- [ ] Headings follow logical hierarchy (h1 → h2 → h3)
- [ ] Tables have proper headers and captions
- [ ] Modal dialogs trap focus and can be closed with Escape key

## ML Platform Specific Accessibility Patterns

### Dataset Upload and File Handling

```jsx
// Good - Accessible file upload with drag and drop
function DatasetUploadZone({ onFileSelect, maxSize, acceptedTypes }) {
  const [isDragOver, setIsDragOver] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileValidation(files);
  };

  const handleFileValidation = (files) => {
    const file = files[0];
    if (!file) return;

    // Announce validation results
    if (file.size > maxSize) {
      setUploadStatus(
        `Error: File size ${formatFileSize(
          file.size
        )} exceeds maximum ${formatFileSize(maxSize)}`
      );
      return;
    }

    if (!acceptedTypes.includes(file.type)) {
      setUploadStatus(
        `Error: File type ${file.type} not supported. Please upload a CSV file.`
      );
      return;
    }

    setUploadStatus(`File ${file.name} selected successfully`);
    onFileSelect(file);
  };

  return (
    <div className="upload-zone">
      <div
        className={`drop-zone ${isDragOver ? "drag-over" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        role="button"
        tabIndex="0"
        aria-describedby="upload-instructions upload-status"
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            fileInputRef.current?.click();
          }
        }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv"
          onChange={(e) => handleFileValidation(Array.from(e.target.files))}
          className="sr-only"
          id="dataset-file-input"
        />

        <div className="upload-content">
          <svg aria-hidden="true" className="upload-icon">
            {/* Upload icon */}
          </svg>
          <p>
            <strong>Click to upload</strong> or drag and drop your CSV file here
          </p>
          <p id="upload-instructions" className="upload-help">
            Maximum file size: {formatFileSize(maxSize)}
          </p>
        </div>
      </div>

      {/* Status announcements */}
      <div
        id="upload-status"
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className={
          uploadStatus.includes("Error") ? "error-message" : "success-message"
        }
      >
        {uploadStatus}
      </div>
    </div>
  );
}
```

### Algorithm Configuration with Dynamic Forms

```jsx
// Good - Accessible dynamic hyperparameter form
function HyperparameterConfig({ algorithm, values, onChange, errors }) {
  const [focusedParam, setFocusedParam] = useState(null);

  const renderParameter = (param) => {
    const hasError = errors[param.name];
    const value = values[param.name] ?? param.default;

    switch (param.type) {
      case "number":
        return (
          <div key={param.name} className="param-group">
            <label htmlFor={`param-${param.name}`}>
              {param.displayName}
              {param.required && (
                <span className="required" aria-label="required">
                  *
                </span>
              )}
            </label>

            <div className="number-input-group">
              <input
                id={`param-${param.name}`}
                type="number"
                min={param.min}
                max={param.max}
                step={param.step}
                value={value}
                onChange={(e) =>
                  onChange(param.name, parseFloat(e.target.value))
                }
                onFocus={() => setFocusedParam(param.name)}
                onBlur={() => setFocusedParam(null)}
                aria-describedby={`${param.name}-help ${
                  hasError ? `${param.name}-error` : ""
                }`}
                aria-invalid={hasError ? "true" : "false"}
                required={param.required}
              />

              {param.showSlider && (
                <input
                  type="range"
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={value}
                  onChange={(e) =>
                    onChange(param.name, parseFloat(e.target.value))
                  }
                  aria-label={`${param.displayName} slider`}
                  className="param-slider"
                />
              )}
            </div>

            <div id={`${param.name}-help`} className="param-help">
              {param.description}
              {param.min !== undefined && param.max !== undefined && (
                <span>
                  {" "}
                  (Range: {param.min} - {param.max})
                </span>
              )}
            </div>

            {hasError && (
              <div
                id={`${param.name}-error`}
                className="error-message"
                role="alert"
              >
                {errors[param.name]}
              </div>
            )}
          </div>
        );

      case "select":
        return (
          <div key={param.name} className="param-group">
            <label htmlFor={`param-${param.name}`}>
              {param.displayName}
              {param.required && (
                <span className="required" aria-label="required">
                  *
                </span>
              )}
            </label>

            <select
              id={`param-${param.name}`}
              value={value}
              onChange={(e) => onChange(param.name, e.target.value)}
              aria-describedby={`${param.name}-help ${
                hasError ? `${param.name}-error` : ""
              }`}
              aria-invalid={hasError ? "true" : "false"}
              required={param.required}
            >
              {param.options.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>

            <div id={`${param.name}-help`} className="param-help">
              {param.description}
            </div>

            {hasError && (
              <div
                id={`${param.name}-error`}
                className="error-message"
                role="alert"
              >
                {errors[param.name]}
              </div>
            )}
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <fieldset className="hyperparameter-config">
      <legend>
        {algorithm.name} Configuration
        <button
          type="button"
          onClick={() => onChange("reset")}
          className="reset-button"
          aria-describedby="reset-help"
        >
          Reset to Defaults
        </button>
      </legend>

      <div id="reset-help" className="sr-only">
        Reset all hyperparameters to their default values
      </div>

      <div className="param-grid">
        {algorithm.parameters.map(renderParameter)}
      </div>

      {/* Live region for parameter explanations */}
      {focusedParam && (
        <div role="status" aria-live="polite" className="param-explanation">
          {
            algorithm.parameters.find((p) => p.name === focusedParam)
              ?.detailedHelp
          }
        </div>
      )}
    </fieldset>
  );
}
```

### Training Progress with Rich Status Updates

```jsx
// Good - Comprehensive training progress with accessibility
function TrainingProgressPanel({ session, onStop }) {
  const [lastAnnouncement, setLastAnnouncement] = useState("");

  // Announce significant progress milestones
  useEffect(() => {
    const progress = session.progress_percentage;
    const status = session.status;

    let announcement = "";

    if (status === "COMPLETED") {
      announcement = `Training completed successfully. Final accuracy: ${session.final_metrics?.accuracy?.toFixed(
        2
      )}%`;
    } else if (status === "FAILED") {
      announcement = `Training failed: ${session.error_message}`;
    } else if (
      progress > 0 &&
      progress % 25 === 0 &&
      progress !== lastAnnouncement
    ) {
      announcement = `Training ${progress}% complete`;
      setLastAnnouncement(progress);
    }

    return announcement;
  }, [session.progress_percentage, session.status]);

  return (
    <section className="training-progress" aria-labelledby="training-heading">
      <header className="progress-header">
        <h2 id="training-heading">Model Training Progress</h2>
        <div className="training-meta">
          <span>Algorithm: {session.algorithm}</span>
          <span>Dataset: {session.dataset_name}</span>
        </div>
      </header>

      {/* Visual progress bar */}
      <div className="progress-container">
        <div
          role="progressbar"
          aria-valuenow={session.progress_percentage}
          aria-valuemin="0"
          aria-valuemax="100"
          aria-labelledby="progress-label"
          aria-describedby="progress-details"
          className="progress-bar"
        >
          <div
            className="progress-fill"
            style={{ width: `${session.progress_percentage}%` }}
          />
        </div>

        <div id="progress-label" className="progress-text">
          {session.progress_percentage.toFixed(1)}% Complete
        </div>
      </div>

      {/* Detailed status information */}
      <div id="progress-details" className="progress-details">
        <dl className="status-list">
          <dt>Status:</dt>
          <dd className={`status-${session.status.toLowerCase()}`}>
            {session.status}
          </dd>

          {session.current_epoch && (
            <>
              <dt>Current Epoch:</dt>
              <dd>
                {session.current_epoch} of {session.total_epochs}
              </dd>
            </>
          )}

          {session.estimated_completion && (
            <>
              <dt>Estimated Completion:</dt>
              <dd>
                <time dateTime={session.estimated_completion}>
                  {formatRelativeTime(session.estimated_completion)}
                </time>
              </dd>
            </>
          )}

          {session.current_metrics && (
            <>
              <dt>Current Accuracy:</dt>
              <dd>{(session.current_metrics.accuracy * 100).toFixed(2)}%</dd>
            </>
          )}
        </dl>
      </div>

      {/* Control buttons */}
      <div className="progress-controls">
        {session.status === "RUNNING" && (
          <button
            onClick={onStop}
            className="stop-button"
            aria-describedby="stop-help"
          >
            Stop Training
          </button>
        )}
        <div id="stop-help" className="sr-only">
          Stop the current training session. Progress will be lost.
        </div>
      </div>

      {/* Live announcements */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {announcement}
      </div>

      {/* Error announcements */}
      {session.error_message && (
        <div role="alert" aria-live="assertive" className="error-announcement">
          Training Error: {session.error_message}
        </div>
      )}
    </section>
  );
}
```

### 1. Perceivable

- Information must be presentable in ways users can perceive
- Provide text alternatives for non-text content
- Ensure sufficient color contrast
- Make content adaptable to different presentations

### 2. Operable

- Interface components must be operable by all users
- Make all functionality keyboard accessible
- Give users enough time to read content
- Don't use content that causes seizures

### 3. Understandable

- Information and UI operation must be understandable
- Make text readable and understandable
- Make content appear and operate predictably
- Help users avoid and correct mistakes

### 4. Robust

- Content must be robust enough for various assistive technologies
- Maximize compatibility with assistive technologies

## Semantic HTML Structure

### Use proper HTML5 semantic elements:

```jsx
// Good - Semantic structure
function MLPlatform() {
  return (
    <main>
      <header>
        <h1>ML Training Platform</h1>
        <nav aria-label="Main navigation">
          <ul>
            <li>
              <a href="#upload">Upload Data</a>
            </li>
            <li>
              <a href="#configure">Configure Model</a>
            </li>
            <li>
              <a href="#results">View Results</a>
            </li>
          </ul>
        </nav>
      </header>

      <section aria-labelledby="upload-heading">
        <h2 id="upload-heading">Dataset Upload</h2>
        <form>{/* Form content */}</form>
      </section>

      <aside aria-labelledby="help-heading">
        <h3 id="help-heading">Help & Tips</h3>
        {/* Help content */}
      </aside>

      <footer>
        <p>&copy; 2024 ML Training Platform</p>
      </footer>
    </main>
  );
}
```

### Heading Hierarchy

- Use headings (h1-h6) in logical order
- Don't skip heading levels
- Use only one h1 per page

```jsx
// Good - Proper heading hierarchy
<main>
  <h1>ML Training Platform</h1>

  <section>
    <h2>Dataset Management</h2>
    <h3>Upload New Dataset</h3>
    <h3>Preview Data</h3>
  </section>

  <section>
    <h2>Model Configuration</h2>
    <h3>Algorithm Selection</h3>
    <h4>Hyperparameters</h4>
  </section>
</main>
```

## Form Accessibility

### Labels and Form Controls

```jsx
// Good - Proper form labeling
function DatasetUploadForm() {
  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="form-group">
        <label htmlFor="dataset-file">
          Dataset File (CSV format, max 100MB)
        </label>
        <input
          id="dataset-file"
          type="file"
          accept=".csv"
          required
          aria-describedby="file-help file-error"
          onChange={handleFileChange}
        />
        <div id="file-help" className="help-text">
          Select a CSV file containing your training data
        </div>
        {error && (
          <div id="file-error" className="error-message" role="alert">
            {error}
          </div>
        )}
      </div>

      <fieldset>
        <legend>Task Type</legend>
        <div className="radio-group">
          <input
            id="classification"
            type="radio"
            name="taskType"
            value="classification"
            checked={taskType === "classification"}
            onChange={handleTaskTypeChange}
          />
          <label htmlFor="classification">Classification</label>
        </div>
        <div className="radio-group">
          <input
            id="regression"
            type="radio"
            name="taskType"
            value="regression"
            checked={taskType === "regression"}
            onChange={handleTaskTypeChange}
          />
          <label htmlFor="regression">Regression</label>
        </div>
      </fieldset>

      <button type="submit" disabled={isUploading}>
        {isUploading ? "Uploading..." : "Upload Dataset"}
      </button>
    </form>
  );
}
```

### Form Validation and Error Handling

```jsx
// Good - Accessible error handling
function HyperparameterForm({ algorithm, onSubmit }) {
  const [errors, setErrors] = useState({});
  const [touched, setTouched] = useState({});

  return (
    <form onSubmit={handleSubmit} noValidate>
      <div className="form-group">
        <label htmlFor="n-estimators">
          Number of Estimators
          <span className="required" aria-label="required">
            *
          </span>
        </label>
        <input
          id="n-estimators"
          type="number"
          min="1"
          max="1000"
          value={nEstimators}
          onChange={handleChange}
          onBlur={handleBlur}
          required
          aria-describedby="n-estimators-help n-estimators-error"
          aria-invalid={errors.nEstimators ? "true" : "false"}
        />
        <div id="n-estimators-help" className="help-text">
          Number of trees in the forest (1-1000)
        </div>
        {errors.nEstimators && (
          <div id="n-estimators-error" className="error-message" role="alert">
            {errors.nEstimators}
          </div>
        )}
      </div>

      {/* Form summary for screen readers */}
      {Object.keys(errors).length > 0 && (
        <div
          className="error-summary"
          role="alert"
          aria-labelledby="error-summary-heading"
        >
          <h3 id="error-summary-heading">
            Please correct the following errors:
          </h3>
          <ul>
            {Object.entries(errors).map(([field, error]) => (
              <li key={field}>
                <a href={`#${field}`}>{error}</a>
              </li>
            ))}
          </ul>
        </div>
      )}
    </form>
  );
}
```

## Interactive Components

### Buttons and Links

```jsx
// Good - Accessible buttons
function TrainingControls({ onStart, onStop, isTraining }) {
  return (
    <div className="training-controls">
      <button
        onClick={onStart}
        disabled={isTraining}
        aria-describedby="start-help"
      >
        {isTraining ? "Training in Progress..." : "Start Training"}
      </button>
      <div id="start-help" className="sr-only">
        Begin model training with current configuration
      </div>

      {isTraining && (
        <button
          onClick={onStop}
          className="stop-button"
          aria-describedby="stop-help"
        >
          Stop Training
        </button>
      )}
      <div id="stop-help" className="sr-only">
        Cancel the current training session
      </div>
    </div>
  );
}

// Good - Accessible links
function ModelDownloadLink({ modelId, filename }) {
  return (
    <a
      href={`/api/models/${modelId}/download`}
      download={filename}
      aria-describedby="download-help"
    >
      Download Model ({filename})
      <span className="sr-only"> - Opens download dialog</span>
    </a>
  );
}
```

### Progress Indicators

```jsx
// Good - Accessible progress indicator
function TrainingProgress({ progress, status, estimatedTime }) {
  return (
    <div className="training-progress">
      <h3>Training Progress</h3>
      <div
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin="0"
        aria-valuemax="100"
        aria-labelledby="progress-label"
        aria-describedby="progress-description"
      >
        <div className="progress-bar">
          <div className="progress-fill" style={{ width: `${progress}%` }} />
        </div>
      </div>
      <div id="progress-label">{progress}% Complete</div>
      <div id="progress-description">
        Status: {status}
        {estimatedTime && ` - Estimated time remaining: ${estimatedTime}`}
      </div>
    </div>
  );
}
```

## Data Tables and Charts

### Accessible Data Tables

```jsx
// Good - Accessible data table
function ModelComparisonTable({ models }) {
  return (
    <div className="table-container">
      <table role="table" aria-labelledby="comparison-heading">
        <caption id="comparison-heading">
          Model Performance Comparison
          <span className="table-summary">
            Comparing {models.length} trained models by accuracy, F1 score, and
            training time
          </span>
        </caption>
        <thead>
          <tr>
            <th scope="col">Model Name</th>
            <th scope="col">Algorithm</th>
            <th scope="col" aria-sort="descending">
              Accuracy
              <button
                className="sort-button"
                onClick={() => handleSort("accuracy")}
                aria-label="Sort by accuracy"
              >
                ↓
              </button>
            </th>
            <th scope="col">F1 Score</th>
            <th scope="col">Training Time</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map((model, index) => (
            <tr key={model.id} className={index === 0 ? "best-model" : ""}>
              <th scope="row">
                {model.name}
                {index === 0 && (
                  <span className="badge" aria-label="Best performing model">
                    Best
                  </span>
                )}
              </th>
              <td>{model.algorithm}</td>
              <td>{(model.accuracy * 100).toFixed(2)}%</td>
              <td>{model.f1Score.toFixed(3)}</td>
              <td>{formatDuration(model.trainingTime)}</td>
              <td>
                <button
                  onClick={() => handleDownload(model.id)}
                  aria-label={`Download ${model.name} model`}
                >
                  Download
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

### Accessible Charts and Visualizations

```jsx
// Good - Accessible chart with alternative text
function MetricsChart({ data, chartType }) {
  const chartDescription = generateChartDescription(data);

  return (
    <div className="chart-container">
      <h3 id="chart-title">Model Performance Metrics</h3>
      <div
        className="chart"
        role="img"
        aria-labelledby="chart-title"
        aria-describedby="chart-description"
      >
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="accuracy" fill="#8884d8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div id="chart-description" className="chart-description">
        {chartDescription}
      </div>

      {/* Data table alternative */}
      <details className="chart-data-table">
        <summary>View chart data as table</summary>
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {data.map((item) => (
              <tr key={item.name}>
                <td>{item.name}</td>
                <td>{(item.accuracy * 100).toFixed(2)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
```

## Keyboard Navigation

### Focus Management

```jsx
// Good - Proper focus management
function AlgorithmSelector({ algorithms, onSelect }) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const listRef = useRef(null);

  const handleKeyDown = (event) => {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault();
        setSelectedIndex((prev) =>
          prev < algorithms.length - 1 ? prev + 1 : 0
        );
        break;
      case "ArrowUp":
        event.preventDefault();
        setSelectedIndex((prev) =>
          prev > 0 ? prev - 1 : algorithms.length - 1
        );
        break;
      case "Enter":
      case " ":
        event.preventDefault();
        onSelect(algorithms[selectedIndex]);
        break;
      case "Escape":
        // Close selector or return focus to trigger
        break;
    }
  };

  return (
    <div className="algorithm-selector">
      <label id="algorithm-label">Select Algorithm:</label>
      <ul
        ref={listRef}
        role="listbox"
        aria-labelledby="algorithm-label"
        aria-activedescendant={`algorithm-${selectedIndex}`}
        tabIndex="0"
        onKeyDown={handleKeyDown}
      >
        {algorithms.map((algorithm, index) => (
          <li
            key={algorithm.id}
            id={`algorithm-${index}`}
            role="option"
            aria-selected={index === selectedIndex}
            className={index === selectedIndex ? "selected" : ""}
            onClick={() => onSelect(algorithm)}
          >
            <strong>{algorithm.name}</strong>
            <p>{algorithm.description}</p>
          </li>
        ))}
      </ul>
    </div>
  );
}
```

### Skip Links

```jsx
// Good - Skip navigation links
function App() {
  return (
    <>
      <a href="#main-content" className="skip-link">
        Skip to main content
      </a>
      <a href="#navigation" className="skip-link">
        Skip to navigation
      </a>

      <header>
        <nav id="navigation" aria-label="Main navigation">
          {/* Navigation content */}
        </nav>
      </header>

      <main id="main-content">{/* Main content */}</main>
    </>
  );
}
```

## ARIA Labels and Descriptions

### Live Regions for Dynamic Content

```jsx
// Good - Live regions for status updates
function TrainingStatus({ status, progress, error }) {
  return (
    <div className="training-status">
      <div aria-live="polite" aria-atomic="true" className="status-updates">
        {status && <p>Status: {status}</p>}
        {progress && <p>Progress: {progress}% complete</p>}
      </div>

      {error && (
        <div role="alert" aria-live="assertive" className="error-message">
          Error: {error}
        </div>
      )}
    </div>
  );
}
```

### Complex UI Components

```jsx
// Good - Accessible modal dialog
function HelpModal({ isOpen, onClose, title, children }) {
  const modalRef = useRef(null);
  const previousFocusRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      previousFocusRef.current = document.activeElement;
      modalRef.current?.focus();
    } else {
      previousFocusRef.current?.focus();
    }
  }, [isOpen]);

  const handleKeyDown = (event) => {
    if (event.key === "Escape") {
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div
      className="modal-overlay"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-labelledby="modal-title"
    >
      <div
        ref={modalRef}
        className="modal-content"
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
        tabIndex="-1"
      >
        <header className="modal-header">
          <h2 id="modal-title">{title}</h2>
          <button
            onClick={onClose}
            aria-label="Close dialog"
            className="close-button"
          >
            ×
          </button>
        </header>
        <div className="modal-body">{children}</div>
      </div>
    </div>
  );
}
```

## Color and Contrast

### Ensure sufficient color contrast:

```css
/* Good - WCAG AA compliant colors */
:root {
  --primary-color: #0066cc; /* 4.5:1 contrast ratio */
  --secondary-color: #004499; /* 7:1 contrast ratio */
  --success-color: #006600; /* 4.5:1 contrast ratio */
  --warning-color: #cc6600; /* 4.5:1 contrast ratio */
  --error-color: #cc0000; /* 5.5:1 contrast ratio */
  --text-color: #333333; /* 12.6:1 contrast ratio */
  --background-color: #ffffff;
}

/* Don't rely solely on color for information */
.status-indicator {
  padding: 0.5rem;
  border-radius: 4px;
}

.status-success {
  background-color: var(--success-color);
  color: white;
}

.status-success::before {
  content: "✓ ";
  font-weight: bold;
}

.status-error {
  background-color: var(--error-color);
  color: white;
}

.status-error::before {
  content: "⚠ ";
  font-weight: bold;
}
```

## Screen Reader Support

### Screen Reader Only Content

```css
/* Screen reader only text */
.sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

/* Show on focus for keyboard users */
.sr-only:focus {
  position: static;
  width: auto;
  height: auto;
  padding: inherit;
  margin: inherit;
  overflow: visible;
  clip: auto;
  white-space: normal;
}
```

### Descriptive Text for Complex Content

```jsx
// Good - Descriptive content for screen readers
function DatasetPreview({ data, columns }) {
  const description = `Dataset contains ${data.length} rows and ${
    columns.length
  } columns. 
    Columns include: ${columns.map((col) => col.name).join(", ")}.`;

  return (
    <div className="dataset-preview">
      <h3>Dataset Preview</h3>
      <p className="sr-only">{description}</p>

      <table aria-label="Dataset preview showing first 5 rows">
        <thead>
          <tr>
            {columns.map((col) => (
              <th key={col.name} scope="col">
                {col.name}
                <span className="sr-only">({col.type} type)</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.slice(0, 5).map((row, index) => (
            <tr key={index}>
              {columns.map((col) => (
                <td key={col.name}>{row[col.name]}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

## Testing Accessibility

### Automated Testing

```javascript
// Use jest-axe for automated accessibility testing
import { render } from "@testing-library/react";
import { axe, toHaveNoViolations } from "jest-axe";

expect.extend(toHaveNoViolations);

test("DatasetUploadForm should be accessible", async () => {
  const { container } = render(<DatasetUploadForm />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### Manual Testing Checklist

- [ ] All interactive elements are keyboard accessible
- [ ] Focus indicators are visible and clear
- [ ] Screen reader announces all important information
- [ ] Color is not the only way to convey information
- [ ] Text has sufficient contrast (4.5:1 for normal text, 3:1 for large text)
- [ ] Images have appropriate alt text
- [ ] Forms have proper labels and error messages
- [ ] Headings are in logical order
- [ ] Live regions announce dynamic content changes

## Tools and Resources

### Browser Extensions

- **axe DevTools** - Automated accessibility testing
- **WAVE** - Web accessibility evaluation
- **Lighthouse** - Includes accessibility audit

### Screen Readers for Testing

- **NVDA** (Windows) - Free screen reader
- **JAWS** (Windows) - Popular commercial screen reader
- **VoiceOver** (macOS) - Built-in screen reader

### Color Contrast Tools

- **WebAIM Contrast Checker**
- **Colour Contrast Analyser**
- **Stark** (Figma/Sketch plugin)

This accessibility guide ensures the ML Training Platform is usable by everyone, regardless of their abilities or the assistive technologies they use.
