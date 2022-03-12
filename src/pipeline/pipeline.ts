import { Matrix } from 'ml-matrix';

export interface PipelineSteps {
  steps?: Array<[string, any]>;
}

export class Pipeline {
  public steps: Array<[string, any]>;

  public constructor({ steps = [] }: PipelineSteps = {}) {
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

  private validateSteps(steps) {}

  public async fit(X: Matrix, y: Matrix) {}

  public transform(X: Matrix) {}

  public predict(X: Matrix) {}
}
