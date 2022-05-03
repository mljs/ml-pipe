import { Matrix } from 'ml-matrix';

import { Estimator } from '../estimators/estimator';
import { Pipeline } from '../pipeline/pipeline';

import { generalizedDegreesOfFreedom } from './complexity';

export function meanSquaredError(y: Matrix, yHat: Matrix): number {
  let sum = 0;

  for (let i = 0; i < y.rows; i++) {
    for (let j = 0; j < y.columns; j++) {
      sum += Math.pow(y.get(i, j) - yHat.get(i, j), 2);
    }
  }

  return sum / (y.rows * y.columns);
}

export function meanAbsoluteError(y: Matrix, yHat: Matrix): number {
  let sum = 0;

  for (let i = 0; i < y.rows; i++) {
    for (let j = 0; j < y.columns; j++) {
      sum += Math.abs(y.get(i, j) - yHat.get(i, j));
    }
  }

  return sum / (y.rows * y.columns);
}
/**
 * DoFAIC is the "generalized degrees of freedom version" of the Akaike Information Criterion.
 * It has been proposed and benchmarked in [1].
 *
 * Similar to the AIC, can be useful in model selection. Based on the train error and the complexity score
 * it can rank models, "trading of Occam's razor with low error".
 *
 * [1]  Gao, Tianxiang,  Vladimir Jojic. ‘Degrees of Freedom in Deep Neural Networks’. arXiv preprint arXiv: Arxiv-1603. 09260 (2016): n. pag. Print.
 * @export
 * @param {Matrix} X
 * @param {Matrix} y
 * @param {(Estimator | Pipeline)} estimator
 * @param {number} [epsilon=1e-5]
 * @param {number} [numberOfDraws=5]
 * @return {*}  {Promise<number>}
 */
export async function doFAIC(
  X: Matrix,
  y: Matrix,
  estimator: Estimator | Pipeline,
  epsilon = 1e-5,
  numberOfDraws = 5,
): Promise<number> {
  const dof = await generalizedDegreesOfFreedom(
    X,
    y,
    estimator,
    epsilon,
    numberOfDraws,
  );
  const predictions = estimator.predict(X);
  const mse = meanSquaredError(y, predictions);
  return mse + 2 * dof;
}
