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

  private validateSteps(steps) {}

  public fit(X: Matrix, y: Matrix) {}

  public transform(X: Matrix) {}

  public predict(X: Matrix) {}
}
