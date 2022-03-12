import { Matrix } from 'ml-matrix';

import { columnWiseMax, columnWiseMin } from '../../utils/colStat';
import { OperationSteps, operationChain } from '../../utils/vectorMatrix';
import { turnZerosToOnes } from '../../utils/zerosToOnes';
import { Transformer } from '../transformer';

export class MinMaxScaler implements Transformer {
  private min: number[];
  private range: number[];

  public constructor() {
    this.min = [];
    this.range = [];
  }

  public fit(X: Matrix) {
    this.min = columnWiseMin(X);
    const max = columnWiseMax(X);

    // we turn zeros to ones to avoid division by zero
    this.range = turnZerosToOnes(max.map((v, i) => v - this.min[i]));
  }

  public transform(X: Matrix) {
    let steps: OperationSteps = {
      steps: [
        [this.min, '-'],
        [this.range, '/'],
      ],
    };
    return operationChain(X, steps);
  }

  public inverseTransform(X: Matrix) {
    let steps: OperationSteps = {
      steps: [
        [this.range, '*'],
        [this.min, '+'],
      ],
    };
    return operationChain(X, steps);
  }

  public fitTransform(X: Matrix) {
    this.fit(X);
    return this.transform(X);
  }
}
