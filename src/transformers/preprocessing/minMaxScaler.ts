import { Matrix } from 'ml-matrix';

import { Transformer } from '../transformer';

export class MinMaxScaler implements Transformer {
  private min: Matrix;
  private max: Matrix;
  private range: Matrix;

  public constructor() {
    this.min = new Matrix([]);
    this.max = new Matrix([]);
    this.range = new Matrix([]);
  }

  public fit(X: Matrix) {
    this.min = new Matrix([X.min('column')]);
    this.max = new Matrix([X.max('column')]);
    this.range = this.max.sub(this.min);
  }

  public transform(X: Matrix) {
    return X.sub(this.min).div(this.range);
  }

  public inverseTransform(X: Matrix) {
    return X.mul(this.range).add(this.min);
  }
}
