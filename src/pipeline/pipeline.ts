export interface PipelineSteps {
  steps?: Array<[string, any]>;
}

export class Pipeline {
  steps: Array<[string, any]>;

  public constructor({ steps = [] }: PipelineSteps = {}) {
    this.steps = steps;
    this.validateSteps(this.steps);
  }
}
