import median from 'ml-array-median';
import { Matrix } from 'ml-matrix';

import { Estimator } from '../Estimator';

export interface DummyRegressorOptions {
  /**Strategy to use for dummy regression.
   * For `mean` we will always predict the mean of the targets.
   * For `median` we will always predict the median of the targets.
   * For `constant` we will always predict the constant specified by `constantValue`.
   * For `quantile` we will always predict the specified quantile of the targets.
   */
  strategy?: 'mean' | 'median' | 'constant' | 'quantile';
  constantValue?: number;
  quantile?: number;
}

/**
 * A dummy regressor that always predicts the same value.
 * This model type is useful as a baseline, e.g. to see what MAE and MSE would be if all predictions were the mean.
 *
 * @export
 * @class DummyRegressor
 * @implements {Estimator}
 */
export class DummyRegressor implements Estimator {
  public fitted: boolean;
  private options: DummyRegressorOptions;
  private prediction: number;
  public constructor(options: DummyRegressorOptions) {
    const { strategy = 'mean', constantValue = 0, quantile = 0.5 } = options;

    this.fitted = false;
    this.prediction = 0;
    this.options = { strategy, constantValue, quantile };
  }

  private _fit(X: Matrix, y: Matrix) {
    switch (this.options.strategy) {
      case 'mean':
        this.prediction = y.mean();
        break;
      case 'median':
        this.prediction = median(y.to1DArray());
        break;
      case 'constant':
        if (this.options.constantValue === undefined) {
          throw new Error('constantValue must be specified');
        }
        this.prediction = this.options.constantValue;
        break;
      case 'quantile':
        throw new Error('quantile strategy not yet implemented');
      default:
        throw new Error('Not implemented');
    }
  }
  public async fit(X: Matrix, y: Matrix) {
    this._fit(X, y);
    this.fitted = true;
  }

  public predict(X: Matrix) {
    let prediction = new Matrix(X.rows, 1);
    for (let i = 0; i < X.rows; i++) {
      prediction.set(i, 0, this.prediction);
    }
    return prediction;
  }
}
