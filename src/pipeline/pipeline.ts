import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimators/estimator';

// Implementation inspired by scikitjs

export class Pipeline {
  public steps: Array<[string, Transformer | Estimator]>;

  public constructor(steps: Array<[string, Transformer | Estimator]>) {
    this.steps = steps;
    this.validateSteps(this.steps);
  }

  // Implementation taken from scikitjs
  private isTransformer(possibleTransformer: any) {
    if (possibleTransformer === 'passthrough') {
      return true;
    }
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
    return this.steps[this.steps.length - 1][1];
  }

  // Implementation taken from scikitjs
  private isEstimator(possibleTransformer: any) {
    if (possibleTransformer === 'passthrough') {
      return true;
    }
    if (typeof possibleTransformer.fit === 'function') {
      return true;
    }
    return false;
  }

  private validateSteps(steps: Array<[string, Transformer | Estimator]>) {
    // The last step must be an estimator and all the others must be transformers
    // otherwise it is not really clear what the user wants to do
    const lastStep = steps[steps.length - 1];
    for (let i = 0; i < steps.length - 2; ++i) {
      const step = steps[i];
      if (!this.isTransformer(step[1])) {
        throw new Error(`Step ${i} should be a transformer but is not.`);
      }
    }
    if (!this.isEstimator(lastStep[1])) {
      throw new Error(`Last step should be an estimator but is not.`);
    }
  }

  public async fit(X: Matrix, y: Matrix) {
    let Xt;
    let yt = this.transform(X, y);
    await this.getLastEstimator().fit(Xt, yt);
    return this;
  }

  public transform(X: Matrix, y?: Matrix) {
    let Xt = X;
    let yt = y;

    for (const step of this.steps.slice(0, -1)) {
      let [name, transformer] = step;
      if (!(name === 'passthrough')) {
        if (transformer.onTarget) {
          yt = transformer.transform(yt);
        } else {
          Xt = transformer.transform(Xt);
        }
      }
    }
    return Xt;
  }

  public predict(X: Matrix, y?: Matrix) {
    let transformed = this.transform(X, y);
    return this.getLastEstimator().predict(transformed[0]);
  }
}
