import { Matrix } from 'ml-matrix';
import MLR from 'ml-regression-multivariate-linear';

import { Estimator } from '../Estimator';

export class LinearRegressor implements Estimator {
  private intercept: boolean;
  private model: MLR | undefined;
  public constructor(intercept?: boolean) {
    this.intercept = intercept || true;
    this.model = undefined;
  }

  public async fit(X: Matrix, y: Matrix) {
    this.model = new MLR(X.to2DArray(), y.to2DArray(), {
      intercept: this.intercept,
    });
  }

  public predict(X: Matrix) {
    if (this.model === undefined) {
      throw new Error('Model not fitted');
    }
    return new Matrix(this.model.predict(X.to2DArray()));
  }
}
