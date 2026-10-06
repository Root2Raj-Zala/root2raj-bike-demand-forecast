"""One-day-ahead bike demand: audited chronological evaluation on genuine UCI data."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import urllib.request
import zipfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE_URL = 'https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip'
FEATURES = ['lag_1', 'lag_7', 'lag_14', 'lag_28', 'mean_7', 'mean_28',
            'weekday_sin', 'weekday_cos', 'year_sin', 'year_cos', 'holiday', 'trend']
FORBIDDEN = {'cnt', 'casual', 'registered', 'temp', 'atemp', 'hum', 'windspeed', 'weathersit'}
ALPHAS = [1.0, 10.0, 100.0]
BASELINES = ['yesterday', 'last_week', 'four_week_weekday_mean']
VALIDATION_END = pd.Timestamp('2012-06-30')
TEST_START = pd.Timestamp('2012-07-01')


def load_source(path=None):
    path = Path(path) if path else ROOT / 'data' / 'day.csv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
        (path.parent / 'source.zip').write_bytes(raw)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            path.write_bytes(archive.read('day.csv'))
    data = pd.read_csv(path, parse_dates=['dteday']).sort_values('dteday').reset_index(drop=True)
    required = {'dteday', 'cnt', 'casual', 'registered', 'holiday', 'weathersit'}
    assert required <= set(data), 'Required source columns missing'
    assert len(data) == 731, 'Unexpected daily source row count'
    assert data.dteday.is_unique, 'Dates must be unique'
    assert data.dteday.tolist() == list(pd.date_range('2011-01-01', '2012-12-31')), 'Dates must be consecutive'
    assert not data[list(required)].isna().any().any(), 'Missing required values'
    assert (data.cnt >= 0).all(), 'Rental counts must be nonnegative'
    assert (data.casual + data.registered == data.cnt).all(), 'Components must reconcile to target'
    return data, hashlib.sha256(path.read_bytes()).hexdigest()


def build_features(data):
    """All count-derived inputs are shifted; date and holiday are known in advance."""
    count = data.cnt.astype(float)
    out = pd.DataFrame(index=data.index)
    for lag in [1, 7, 14, 28]:
        out[f'lag_{lag}'] = count.shift(lag)
    for window in [7, 28]:
        out[f'mean_{window}'] = count.shift(1).rolling(window).mean()
    weekday = data.dteday.dt.dayofweek
    dayofyear = data.dteday.dt.dayofyear
    out['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    out['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    out['year_sin'] = np.sin(2 * np.pi * dayofyear / 365.25)
    out['year_cos'] = np.cos(2 * np.pi * dayofyear / 365.25)
    out['holiday'] = data.holiday.astype(float)
    out['trend'] = (data.dteday - pd.Timestamp('2011-01-01')).dt.days.astype(float)
    return out[FEATURES]


def ridge_predict(x_train, y_train, x_next, alpha):
    """Training-only centering/scaling, unpenalized intercept, nonnegative count."""
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    x = (x_train - mean) / scale
    target_mean = y_train.mean()
    coefficients = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ (y_train - target_mean))
    return max(0.0, float(target_mean + ((x_next - mean) / scale) @ coefficients))


def score(actual, prediction):
    error = np.asarray(prediction) - np.asarray(actual)
    return {'mae_rentals': float(np.abs(error).mean()),
            'rmse_rentals': float(np.sqrt((error ** 2).mean())),
            'wape_percent': float(100 * np.abs(error).sum() / np.asarray(actual).sum()),
            'bias_rentals': float(error.mean())}


def walk_forward(data, features):
    """After each day's actual is observed, refit and forecast only the next day."""
    rows = []
    audits = []
    first_complete = int(features.dropna().index.min())
    for idx in data.index[data.dteday >= pd.Timestamp('2012-01-01')]:
        start = max(first_complete, idx - 365)
        train_index = np.arange(start, idx)
        x = features.loc[train_index].to_numpy(float)
        y = data.loc[train_index, 'cnt'].to_numpy(float)
        row = {'date': data.loc[idx, 'dteday'].strftime('%Y-%m-%d'),
               'actual': int(data.loc[idx, 'cnt']),
               'period': 'validation' if data.loc[idx, 'dteday'] <= VALIDATION_END else 'test',
               'holiday': int(data.loc[idx, 'holiday']),
               'weather_observed_after_forecast': int(data.loc[idx, 'weathersit'])}
        row['yesterday'] = float(data.loc[idx - 1, 'cnt'])
        row['last_week'] = float(data.loc[idx - 7, 'cnt'])
        row['four_week_weekday_mean'] = float(data.loc[[idx - 7, idx - 14, idx - 21, idx - 28], 'cnt'].mean())
        for alpha in ALPHAS:
            row[f'ridge_{alpha:g}'] = ridge_predict(x, y, features.loc[idx].to_numpy(float), alpha)
        rows.append(row)
        audits.append({'forecast_date': row['date'],
                       'latest_observed_count_date': data.loc[idx - 1, 'dteday'].strftime('%Y-%m-%d'),
                       'training_first_date': data.loc[start, 'dteday'].strftime('%Y-%m-%d'),
                       'training_last_date': data.loc[idx - 1, 'dteday'].strftime('%Y-%m-%d'),
                       'training_rows': int(len(train_index))})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def fit_and_evaluate(data, features):
    predictions, audit = walk_forward(data, features)
    methods = BASELINES + [f'ridge_{a:g}' for a in ALPHAS]
    validation = predictions[predictions.period == 'validation']
    test = predictions[predictions.period == 'test']
    validation_scores = {m: score(validation.actual, validation[m]) for m in methods}
    selected = min(methods, key=lambda m: validation_scores[m]['mae_rentals'])
    best_baseline = min(BASELINES, key=lambda m: validation_scores[m]['mae_rentals'])
    test_scores = {m: score(test.actual, test[m]) for m in methods}
    # Freeze the method on validation. Test outcomes never change the selection.
    predictions['selected_forecast'] = predictions[selected]
    predictions['absolute_error'] = (predictions.selected_forecast - predictions.actual).abs()
    residuals = validation.actual.to_numpy() - validation[selected].to_numpy()
    # Fixed descriptive interval from validation residuals; coverage is measured, not promised.
    lower, upper = np.quantile(residuals, [0.05, 0.95], method='linear')
    predictions['interval_lower'] = (predictions.selected_forecast + lower).clip(lower=0)
    predictions['interval_upper'] = (predictions.selected_forecast + upper).clip(lower=0)
    test = predictions[predictions.period == 'test'].copy()
    coverage = float(100 * ((test.actual >= test.interval_lower) & (test.actual <= test.interval_upper)).mean())
    monthly = []
    test['month'] = pd.to_datetime(test.date).dt.strftime('%Y-%m')
    for month, group in test.groupby('month'):
        monthly.append({'month': month, 'days': len(group), **score(group.actual, group.selected_forecast)})
    weather = []
    for label, group in test.groupby('weather_observed_after_forecast'):
        weather.append({'observed_weather_code': int(label), 'days': len(group),
                        **score(group.actual, group.selected_forecast)})
    summary = {'source': {'name': 'UCI Bike Sharing, daily file', 'url': SOURCE_URL,
                         'landing_page': 'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
                         'citation': 'Fanaee-T, H. (2013). Bike Sharing [Dataset]. DOI: 10.24432/C5W894.',
                         'license': 'CC BY 4.0', 'rows': len(data),
                         'start': '2011-01-01', 'end': '2012-12-31'},
               'protocol': {'horizon': 'one day ahead; previous daily total assumed available',
                            'initial_training': '2011-01-29 to 2011-12-31 (lags need 28 warm-up days)',
                            'validation': '2012-01-01 to 2012-06-30', 'validation_days': len(validation),
                            'test': '2012-07-01 to 2012-12-31', 'test_days': len(test),
                            'max_training_window_days': 365, 'refit_frequency': 'daily',
                            'selection_rule': 'lowest validation MAE; frozen before test',
                            'features': FEATURES, 'excluded_same_day_inputs': sorted(FORBIDDEN),
                            'ridge_penalties_predeclared': ALPHAS},
               'selected_method': selected, 'validation_selected_baseline': best_baseline,
               'validation_scores': validation_scores, 'test_scores': test_scores,
               'test_mae_reduction_vs_validation_selected_baseline_percent':
                    float(100 * (1 - test_scores[selected]['mae_rentals'] / test_scores[best_baseline]['mae_rentals'])),
               'test_interval_coverage_percent': coverage,
               'interval_method': 'validation residual 5th/95th quantiles; fixed during test; descriptive, not guaranteed',
               'test_monthly': monthly, 'test_retrospective_weather_error': weather,
               'worst_test_days': test.nlargest(5, 'absolute_error')[['date', 'actual', 'selected_forecast', 'absolute_error']].to_dict('records')}
    return summary, predictions, audit


def save_charts(predictions, summary):
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache' / 'matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['svg.fonttype'] = 'none'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': '#faf8f3', 'axes.facecolor': '#faf8f3',
                         'savefig.facecolor': '#faf8f3', 'svg.hashsalt': 'root2raj-bike-v1'})
    test = predictions[predictions.period == 'test']
    date = pd.to_datetime(test.date)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    axes[0].fill_between(date, test.interval_lower, test.interval_upper, color='#ce8051', alpha=.18, label='Validation residual band (nominal 90%)')
    axes[0].plot(date, test.actual, color='#172f46', linewidth=1.4, label='Actual recorded rentals')
    axes[0].plot(date, test.selected_forecast, color='#bf5c24', linewidth=1.1, label='One-day-ahead forecast')
    axes[0].set_ylabel('Rentals per day')
    axes[0].set_title('Can yesterday\'s evidence predict tomorrow?  |  Held-out July–December 2012', loc='left', pad=17, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[1].plot(date, test.selected_forecast - test.actual, color='#bf5c24', linewidth=.8)
    axes[1].axhline(0, color='#172f46', linewidth=.6)
    axes[1].set_ylabel('Forecast − actual')
    fig.autofmt_xdate()
    fig.text(.07, .02, 'Source: Fanaee-T (2013), UCI Bike Sharing. Daily rolling refits; historical case study, not an operational deployment.', fontsize=9, color='#536475')
    fig.tight_layout(rect=(0, .05, 1, 1))
    for extension in ['svg', 'png']:
        fig.savefig(ROOT / 'outputs' / f'heldout_forecast.{extension}', dpi=150, metadata={'Date': None} if extension == 'svg' else None)
    plt.close(fig)
    methods = sorted(summary['test_scores'], key=lambda m: summary['test_scores'][m]['mae_rentals'])
    fig, ax = plt.subplots(figsize=(10, 4.7))
    colors = ['#bf5c24' if m == summary['selected_method'] else '#637b8d' for m in methods]
    bars = ax.barh(methods, [summary['test_scores'][m]['mae_rentals'] for m in methods], color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, fmt='%.0f', padding=7)
    ax.set_xlim(0, max(b.get_width() for b in bars) * 1.15)
    ax.set_xlabel('Mean absolute error, rentals/day (lower is better)')
    ax.set_title('Held-out results: the simple baselines must earn their place', loc='left', pad=17, fontweight='bold')
    fig.text(.05, .025, 'Orange = method selected on January–June validation. Test results are reported for comparison, not reselection.', fontsize=9, color='#536475')
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(ROOT / 'outputs' / 'model_comparison.svg', metadata={'Date': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-charts', action='store_true', help='Run the data/model checks without plotting')
    args = parser.parse_args()
    data, sha = load_source()
    features = build_features(data)
    summary, predictions, audit = fit_and_evaluate(data, features)
    summary['source']['sha256_day_csv'] = sha
    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    summary_path = output / 'metrics.json'
    if summary_path.exists():
        # The held-out result is regression-checked; no tuning against test metrics.
        previous = json.loads(summary_path.read_text(encoding='utf-8'))
        if previous.get('source', {}).get('sha256_day_csv') == sha:
            assert previous['selected_method'] == summary['selected_method'], 'Selection changed'
            assert np.isclose(previous['test_scores'][summary['selected_method']]['mae_rentals'],
                              summary['test_scores'][summary['selected_method']]['mae_rentals']), 'Metrics changed'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    predictions.to_csv(output / 'predictions.csv', index=False, float_format='%.6f')
    audit.to_csv(output / 'forecast_audit.csv', index=False)
    validation = {'source_731_consecutive_days': True, 'rental_components_reconcile': True,
                  'no_same_day_outcome_or_observed_weather_features': not bool(set(FEATURES) & FORBIDDEN),
                  'every_training_cutoff_precedes_forecast': bool((pd.to_datetime(audit.training_last_date) < pd.to_datetime(audit.forecast_date)).all()),
                  'chronological_validation_test_split': bool(summary['protocol']['validation_days'] == 182 and summary['protocol']['test_days'] == 184),
                  'forecasts_finite_and_nonnegative': bool(np.isfinite(predictions.selected_forecast).all() and (predictions.selected_forecast >= 0).all()),
                  'feature_shift_matches_previous_day': bool(np.allclose(features.lag_1.iloc[28:], data.cnt.iloc[27:-1])),
                  'model_selection_uses_validation_only': summary['selected_method'] == min(summary['validation_scores'], key=lambda m: summary['validation_scores'][m]['mae_rentals'])}
    assert all(validation.values()), validation
    (output / 'validation.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
    if not args.no_charts:
        save_charts(predictions, summary)
    print(json.dumps({'selected_method': summary['selected_method'],
                      'test': summary['test_scores'][summary['selected_method']],
                      'baseline': summary['validation_selected_baseline'],
                      'mae_reduction_percent': summary['test_mae_reduction_vs_validation_selected_baseline_percent'],
                      'interval_coverage_percent': summary['test_interval_coverage_percent'],
                      'checks_passed': sum(validation.values())}, indent=2))

if __name__ == '__main__':
    main()
