import { Matrix } from 'ml-matrix';
import { RandomForestRegression } from 'ml-random-forest';

import { Estimator } from '../estimator';

export class RandomForestRegressor implements Estimator {
  private options: any;
  private regressor: RandomForestRegression;
  public constructor(options: any = {}) {
    this.options = options;
    this.regressor = new RandomForestRegression(options);
  }

  public async fit(X: Matrix, y: Matrix) {
    this.regressor.train(X.to2DArray(), y.to1DArray());
  }

  public predict(X: Matrix) {
    return Matrix.from1DArray(X.rows, 1, this.regressor.predict(X.to2DArray()));
  }
}
