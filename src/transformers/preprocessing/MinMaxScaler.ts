import { Matrix } from 'ml-matrix';

import { OperationSteps, operationChain } from '../../utils/vectorMatrix';
import { turnZerosToOnes } from '../../utils/zerosToOnes';
import { Transformer } from '../Transformer';

export class MinMaxScaler implements Transformer {
  private min: number[];
  private range: number[];
  public onTarget: boolean;
  public fitted: boolean;

  public constructor(onTarget?: boolean) {
    this.onTarget = onTarget || false;
    this.min = [];
    this.range = [];
    this.fitted = false;
  }

  public fit(X: Matrix) {
    this.min = X.min('column');
    const max = X.max('column');

    // we turn zeros to ones to avoid division by zero
    this.range = turnZerosToOnes(max.map((v, i) => v - this.min[i]));
    this.fitted = true;
  }

  public transform(X: Matrix) {
    if (!this.fitted) {
      throw new Error('You must fit the transformer before using it');
    }
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

export class TargetMinMaxScaler extends MinMaxScaler {
  public onTarget: boolean;

  public constructor(onTarget?: boolean) {
    super(onTarget);
    this.onTarget = true;
  }
}
