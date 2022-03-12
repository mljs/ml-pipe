import { Matrix } from 'ml-matrix';

import { operationChain, OperationSteps } from '../vectorMatrix';

describe('test the operatorchain', () => {
  it('simple matrix additions', () => {
    const m = Matrix.zeros(2, 2);
    const vector = [1, 2];
    const steps: OperationSteps = {
      steps: [
        [vector, '+'],
        [vector, '+'],
      ],
    };
    const result = operationChain(m, steps);
    expect(result.to2DArray()).toStrictEqual([
      [2, 4],
      [2, 4],
    ]);
  });

  it('simple matrix addition and substraction', () => {
    const m = Matrix.zeros(2, 2);
    const vector = [1, 2];
    const steps: OperationSteps = {
      steps: [
        [vector, '+'],
        [vector, '-'],
      ],
    };
    const result = operationChain(m, steps);
    expect(result.to2DArray()).toStrictEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it('testing division and multiplication', () => {
    const m = Matrix.zeros(2, 2);
    const vector = [1, 2];
    const steps: OperationSteps = {
      steps: [
        [vector, '*'],
        [vector, '/'],
      ],
    };
    const result = operationChain(m, steps);
    expect(result.to2DArray()).toStrictEqual([
      [0, 0],
      [0, 0],
    ]);
  });

  it('only one column', () => {
    const m = Matrix.zeros(2, 1);
    const vector = [1];
    const steps: OperationSteps = {
      steps: [
        [vector, '*'],
        [vector, '/'],
      ],
    };
    const result = operationChain(m, steps);
    expect(result.to2DArray()).toStrictEqual([[0], [0]]);
  });
});
