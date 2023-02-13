import { Matrix } from 'ml-matrix';

import { OperationSteps, operationChain } from '../../utils/vectorMatrix';
import { turnZerosToOnes } from '../../utils/zerosToOnes';
import { Transformer } from '../Transformer';

export class StandardScaler implements Transformer {
  private std: number[];
  private mean: number[];
  public onTarget: boolean;
  public fitted: boolean;

  public constructor(onTarget?: boolean) {
    this.onTarget = onTarget || false;
    this.std = [];
    this.mean = [];
    this.fitted = false;
  }

  public fit(X: Matrix) {
    // using biased estimator following sklearn convention
    // we turn zeros to ones to avoid division by zero
    this.std = turnZerosToOnes(
      X.standardDeviation('column', { unbiased: false }),
    );
    this.mean = X.mean('column');
    this.fitted = true;
  }

  public transform(X: Matrix) {
    if (!this.fitted) {
      throw new Error('You must fit the transformer before using it');
    }
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
