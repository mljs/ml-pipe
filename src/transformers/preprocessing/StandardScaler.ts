import { Matrix } from 'ml-matrix';

import { OperationSteps, operationChain } from '../../utils/vectorMatrix';
import { turnZerosToOnes } from '../../utils/zerosToOnes';
import { Transformer } from '../Transformer';

export class StandardScaler implements Transformer {
  private std: number[];
  private mean: number[];
  public onTarget: boolean;

  public constructor(onTarget?: boolean) {
    this.onTarget = onTarget || false;
    this.std = [];
    this.mean = [];
  }

  public fit(X: Matrix) {
    // using biased estimator following sklearn convention
    // we turn zeros to ones to avoid division by zero
    this.std = turnZerosToOnes(
      X.standardDeviation('column', { unbiased: false }),
    );
    this.mean = X.mean('column');
  }

  public transform(X: Matrix) {
    let steps: OperationSteps = {
      steps: [
        [this.mean, '-'],
        [this.std, '/'],
      ],
    };
    return operationChain(X, steps);
  }

  public inverseTransform(X: Matrix) {
    let steps: OperationSteps = {
      steps: [
        [this.std, '*'],
        [this.mean, '+'],
      ],
    };
    return operationChain(X, steps);
  }

  public fitTransform(X: Matrix) {
    this.fit(X);
    return this.transform(X);
  }
}

export class TargetStandardScaler extends StandardScaler {
  public onTarget: boolean;

  public constructor(onTarget?: boolean) {
    super(onTarget);
    this.onTarget = true;
  }
}
