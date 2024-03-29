import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimators/Estimator';
import { Transformer } from '../transformers/Transformer';
// Implementation inspired by scikitjs

export interface Step {
  name: string;
  object: Transformer | Estimator;
}

export class Pipeline {
  public steps: Array<Step>;

  public constructor(steps: Array<Step>) {
    this.steps = steps;
    this.validateSteps(this.steps);
  }

  // Implementation taken from scikitjs
  private isTransformer(possibleTransformer: any) {
    if (
      typeof possibleTransformer.fit === 'function' &&
      typeof possibleTransformer.transform === 'function' &&
      typeof possibleTransformer.fitTransform === 'function'
    ) {
      return true;
    }
    return false;
  }

  private getLastEstimator() {
    return this.steps[this.steps.length - 1].object;
  }

  // Implementation taken from scikitjs
  private isEstimator(possibleEstimator: any) {
    if (
      typeof possibleEstimator.fit === 'function' &&
      typeof possibleEstimator.predict === 'function'
    ) {
      return true;
    }
    return false;
  }

  private validateSteps(steps: Array<Step>) {
    // The last step must be an estimator and all the others must be transformers
    // otherwise it is not really clear what the user wants to do
    const lastStep = steps[steps.length - 1];
    if (steps.length > 1) {
      for (let i = 0; i < steps.length - 1; ++i) {
        const step = steps[i];
        if (!this.isTransformer(step.object)) {
          throw new Error(`Step ${i} should be a transformer but is not.`);
        }
      }
    }
    if (!this.isEstimator(lastStep.object)) {
      throw new Error(`Last step should be an estimator but is not.`);
    }
  }

  public async fit(X: Matrix, y: Matrix) {
    let Xt = X;
    let yt = y;
    for (const step of this.steps.slice(0, -1)) {
      let { name, object: transformer } = step;
      if ('transform' in transformer) {
        if (!(name === 'passthrough')) {
          if (transformer.onTarget) {
            if (yt === undefined) {
              throw new Error('y is undefined');
            }
            yt = transformer.fitTransform(yt);
          } else {
            Xt = transformer.fitTransform(Xt);
          }
        }
      }
    }
    await this.fitLastEstimator(Xt, yt);
  }

  private async fitLastEstimator(x: Matrix, y: Matrix) {
    const lastEstimator = this.getLastEstimator();
    if ('fit' in lastEstimator) {
      await lastEstimator.fit(x, y);
    } else {
      throw new Error('last step of the pipeline is not an estimator');
    }
    return this;
  }

  public transform(X: Matrix, y?: Matrix) {
    let Xt = X;
    let yt = y;
    for (const step of this.steps.slice(0, -1)) {
      let { name, object: transformer } = step;
      if ('transform' in transformer) {
        if (!(name === 'passthrough')) {
          if (transformer.onTarget) {
            if (yt !== undefined) {
              yt = transformer.transform(yt);
            }
          } else {
            Xt = transformer.transform(Xt);
          }
        }
      }
    }
    return { Xt, yt };
  }

  public predict(X: Matrix, y?: Matrix) {
    let { Xt } = this.transform(X, y);

    let prediction;
    const lastEstimator = this.getLastEstimator();
    if ('predict' in lastEstimator) {
      prediction = lastEstimator.predict(Xt);
    } else {
      throw new Error('last step of the pipeline is not an estimator');
    }

    for (const step of this.steps.slice(0, -1)) {
      let { name, object: transformer } = step;
      if ('transform' in transformer) {
        if (!(name === 'passthrough')) {
          if (transformer.onTarget) {
            prediction = transformer.inverseTransform(prediction);
          }
        }
      }
    }
    return prediction;
  }
}
