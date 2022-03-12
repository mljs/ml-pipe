import { Matrix } from 'ml-matrix';

export const operations = {
  '+': (arg0: number, arg1: number) => arg0 + arg1,
  '-': (arg0: number, arg1: number) => arg0 - arg1,
  '*': (arg0: number, arg1: number) => arg0 * arg1,
  '/': (arg0: number, arg1: number) => arg0 / arg1,
};

export interface OperationSteps {
  steps: Array<[number[], '+' | '-' | '*' | '/']>;
}

function getOperation(
  operation: '+' | '-' | '*' | '/',
): (arg0: number, arg1: number) => number {
  return operations[operation];
}

function computeOperation(
  arg0: number,
  arg1: number,
  operation: '+' | '-' | '*' | '/',
): number {
  const operationFunc = getOperation(operation);
  return operationFunc(arg0, arg1);
}

export function operationChain(x: Matrix, steps: OperationSteps) {
  const rows = x.rows;
  const cols = x.columns;
  let output = Matrix.zeros(rows, cols);

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      let res = x.get(i, j);
      for (const [arg0, operation] of steps.steps) {
        res = computeOperation(res, arg0[j], operation);
      }
      output.set(i, j, res);
    }
  }

  return new Matrix(output);
}
